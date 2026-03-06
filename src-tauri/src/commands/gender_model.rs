use crate::gender_detector::GenderDetector;
use futures_util::StreamExt;
use log::info;
use serde::{Deserialize, Serialize};
use specta::Type;
use std::io::Write;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

const GENDER_MODEL_URL: &str = "https://huggingface.co/prithivMLmods/Common-Voice-Gender-Detection-ONNX/resolve/main/onnx/model.onnx";
const GENDER_MODEL_FILENAME: &str = "gender_detection.onnx";

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct GenderModelStatus {
    pub is_downloaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct GenderModelDownloadProgress {
    pub downloaded: u64,
    pub total: u64,
    pub percentage: f64,
}

pub fn gender_model_path(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    app.path()
        .app_data_dir()
        .map_err(|e| e.to_string())
        .map(|d| d.join("models").join(GENDER_MODEL_FILENAME))
}

#[tauri::command]
#[specta::specta]
pub fn get_gender_model_status(app: AppHandle) -> Result<GenderModelStatus, String> {
    let path = gender_model_path(&app)?;
    Ok(GenderModelStatus {
        is_downloaded: path.exists(),
    })
}

#[tauri::command]
#[specta::specta]
pub async fn download_gender_model(app: AppHandle) -> Result<(), String> {
    let path = gender_model_path(&app)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .get(GENDER_MODEL_URL)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;

    let total = response.content_length().unwrap_or(0);
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();

    let tmp_path = path.with_extension("onnx.tmp");
    let mut file = std::fs::File::create(&tmp_path).map_err(|e| e.to_string())?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| e.to_string())?;
        file.write_all(&chunk).map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;

        let percentage = if total > 0 {
            downloaded as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        let _ = app.emit(
            "gender-model-progress",
            GenderModelDownloadProgress {
                downloaded,
                total,
                percentage,
            },
        );
    }

    drop(file);
    std::fs::rename(&tmp_path, &path).map_err(|e| e.to_string())?;
    info!("Gender model downloaded to {:?}", path);

    // Load immediately into managed state so it's usable without a restart
    let detector_state = app.state::<Arc<Mutex<Option<GenderDetector>>>>();
    match GenderDetector::new(path.to_str().unwrap()) {
        Ok(detector) => {
            *detector_state.lock().unwrap() = Some(detector);
            let _ = app.emit("gender-model-ready", ());
            info!("Gender model loaded into memory");
        }
        Err(e) => {
            return Err(format!("Model downloaded but failed to load: {}", e));
        }
    }

    Ok(())
}
