use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use timesfm3::checkpoint::Safetensors;
use timesfm3::config::{ModelConfig, QuantizationPrecision};
use timesfm3::json::Json;
use timesfm3::model::timesfm::TimesFM3;

pub struct LoadedModel {
    pub ckpt_path: PathBuf,
    pub precision: QuantizationPrecision,
    pub model: TimesFM3,
}

fn resolve_ckpt_dir<P: AsRef<Path>>(dir: P) -> PathBuf {
    let p = dir.as_ref();
    if p.is_file() {
        if let Some(parent) = p.parent() {
            if parent.join("model.safetensors").exists() {
                return parent.to_path_buf();
            }
        }
    }
    if p.join("model.safetensors").exists() {
        return p.to_path_buf();
    }
    if let Ok(cwd) = std::env::current_dir() {
        let in_cwd = cwd.join(p);
        if in_cwd.join("model.safetensors").exists() {
            return in_cwd;
        }
        if let Some(parent) = cwd.parent() {
            let in_parent = parent.join(p);
            if in_parent.join("model.safetensors").exists() {
                return in_parent;
            }
            if let Some(grand) = parent.parent() {
                let in_grand = grand.join(p);
                if in_grand.join("model.safetensors").exists() {
                    return in_grand;
                }
            }
        }
    }
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let in_exe = exe_dir.join(p);
            if in_exe.join("model.safetensors").exists() {
                return in_exe;
            }
        }
    }
    p.to_path_buf()
}

#[derive(Default, Clone)]
pub struct AppState {
    pub current_model: Arc<Mutex<Option<LoadedModel>>>,
}

impl AppState {
    pub fn get_or_load_model<P: AsRef<Path>>(
        &self,
        ckpt_dir: P,
        precision_override: Option<QuantizationPrecision>,
    ) -> Result<Arc<TimesFM3>, String> {
        let ckpt_path = resolve_ckpt_dir(ckpt_dir);
        let mut guard = self.current_model.lock().map_err(|e| e.to_string())?;

        // Check if matching model is already loaded
        if let Some(ref loaded) = *guard {
            if loaded.ckpt_path == ckpt_path {
                if let Some(p) = precision_override {
                    if loaded.precision == p {
                        return Ok(Arc::new(loaded.model.clone()));
                    }
                } else {
                    return Ok(Arc::new(loaded.model.clone()));
                }
            }
        }

        // Need to load model from disk
        let cfg_path = ckpt_path.join("config.json");
        let mut config = ModelConfig::default();
        if cfg_path.exists() {
            let cfg_bytes = std::fs::read(&cfg_path)
                .map_err(|e| format!("读取 config.json 失败: {e}"))?;
            let root = Json::parse(&cfg_bytes)
                .map_err(|e| format!("解析 config.json 失败: {e}"))?;
            config.from_config_json(&root)
                .map_err(|e| format!("加载模型配置失败: {e}"))?;
        }
        if let Some(p) = precision_override {
            config.precision = p;
        }
        config.validate().map_err(|e| format!("配置校验失败: {e}"))?;

        let st_path = ckpt_path.join("model.safetensors");
        if !st_path.exists() {
            return Err(format!("在 {} 未找到 model.safetensors 权重文件", ckpt_path.display()));
        }

        let store = Safetensors::load(&st_path)
            .map_err(|e| format!("加载 safetensors 权重失败: {e}"))?;
        let model = TimesFM3::load(&store, &config)
            .map_err(|e| format!("装配模型失败: {e}"))?;

        let precision = config.precision;
        *guard = Some(LoadedModel {
            ckpt_path,
            precision,
            model: model.clone(),
        });

        Ok(Arc::new(model))
    }
}
