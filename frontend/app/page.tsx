"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type Detection = {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox_xyxy: number[];
};

type ImageResponse = {
  model_id: string;
  runtime: string;
  artifact: string;
  image_width: number;
  image_height: number;
  latency_ms: number;
  detections: Detection[];
};

type VideoResponse = {
  model_id: string;
  runtime: string;
  artifact: string;
  frames_processed: number;
  average_latency_ms: number;
  total_video_ms: number;
  frame_results: {
    frame_index: number;
    timestamp_ms: number;
    latency_ms: number;
    detections: Detection[];
  }[];
};

type AvailableModels = Record<string, Record<string, string[]>>;
type RunSummary = {
  label: string;
  artifact: string;
  latencyMs: number;
  detections: number;
};

type VideoClassSummary = {
  className: string;
  count: number;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const fallbackModels: AvailableModels = {
  yolov8n: {
    pytorch: ["yolov8n.pt"]
  }
};

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [modelId, setModelId] = useState("yolov8n");
  const [runtime, setRuntime] = useState("pytorch");
  const [response, setResponse] = useState<ImageResponse | null>(null);
  const [videoResponse, setVideoResponse] = useState<VideoResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [availableModels, setAvailableModels] = useState<AvailableModels>(fallbackModels);
  const [availabilityError, setAvailabilityError] = useState("");
  const [confidence, setConfidence] = useState("0.25");
  const [iou, setIou] = useState("0.45");
  const [imgsz, setImgsz] = useState("640");
  const [runHistory, setRunHistory] = useState<RunSummary[]>([]);

  const isImage = useMemo(() => file?.type.startsWith("image/") ?? false, [file]);
  const modelOptions = useMemo(() => Object.keys(availableModels), [availableModels]);
  const runtimeOptions = useMemo(
    () => Object.keys(availableModels[modelId] ?? {}),
    [availableModels, modelId]
  );
  const selectedArtifact = availableModels[modelId]?.[runtime]?.[0] ?? "Not available";
  const activeLatency = response?.latency_ms ?? videoResponse?.average_latency_ms ?? 0;
  const activeDetections =
    response?.detections.length ??
    videoResponse?.frame_results.reduce((sum, frame) => sum + frame.detections.length, 0) ??
    0;
  const activeMode = file?.type.startsWith("video/") ? "Video" : "Image";
  const previewLabel = file?.name ?? "No file selected";
  const totalVideoDetections = useMemo(
    () =>
      videoResponse?.frame_results.reduce((sum, frame) => sum + frame.detections.length, 0) ?? 0,
    [videoResponse]
  );
  const averageDetectionsPerFrame = useMemo(
    () =>
      videoResponse && videoResponse.frames_processed > 0
        ? totalVideoDetections / videoResponse.frames_processed
        : 0,
    [totalVideoDetections, videoResponse]
  );
  const videoClassBreakdown = useMemo<VideoClassSummary[]>(() => {
    if (!videoResponse) {
      return [];
    }
    const counts = new Map<string, number>();
    for (const frame of videoResponse.frame_results) {
      for (const detection of frame.detections) {
        counts.set(detection.class_name, (counts.get(detection.class_name) ?? 0) + 1);
      }
    }
    return [...counts.entries()]
      .map(([className, count]) => ({ className, count }))
      .sort((left, right) => right.count - left.count)
      .slice(0, 6);
  }, [videoResponse]);

  useEffect(() => {
    async function loadAvailableModels() {
      try {
        const res = await fetch(`${API_BASE_URL}/models`);
        if (!res.ok) {
          throw new Error("Unable to load available model artifacts from backend.");
        }
        const payload = (await res.json()) as AvailableModels;
        if (Object.keys(payload).length === 0) {
          setAvailabilityError("No exported model artifacts found. Only configured defaults may work.");
          setAvailableModels(fallbackModels);
          return;
        }
        setAvailableModels(payload);
        setAvailabilityError("");
      } catch {
        setAvailabilityError("Backend model discovery failed. Start the backend and refresh.");
        setAvailableModels(fallbackModels);
      }
    }

    void loadAvailableModels();
  }, []);

  useEffect(() => {
    if (!modelOptions.includes(modelId) && modelOptions.length > 0) {
      setModelId(modelOptions[0]);
    }
  }, [modelId, modelOptions]);

  useEffect(() => {
    if (!runtimeOptions.includes(runtime) && runtimeOptions.length > 0) {
      setRuntime(runtimeOptions[0]);
    }
  }, [runtime, runtimeOptions]);

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setResponse(null);
    setVideoResponse(null);
    setError("");
    if (!nextFile) {
      setPreviewUrl("");
      return;
    }
    setPreviewUrl(URL.createObjectURL(nextFile));
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("Choose an image or video first.");
      return;
    }

    const endpoint = file.type.startsWith("video/") ? "/detect/video" : "/detect/image";
    const body = new FormData();
    body.append("file", file);
    body.append("model_id", modelId);
    body.append("runtime", runtime);
    body.append("imgsz", imgsz);
    body.append("conf", confidence);
    body.append("iou", iou);
    if (endpoint.endsWith("video")) {
      body.append("sample_stride", "8");
      body.append("max_frames", "24");
    }

    setIsSubmitting(true);
    setError("");
    setResponse(null);
    setVideoResponse(null);

    try {
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        body
      });
      const payload = await res.json();
      if (!res.ok) {
        setError(payload.detail ?? "Inference request failed.");
        return;
      }
      if (endpoint.endsWith("image")) {
        const imagePayload = payload as ImageResponse;
        setResponse(imagePayload);
        setRunHistory((current) => [
          {
            label: `${imagePayload.model_id} · ${imagePayload.runtime}`,
            artifact: imagePayload.artifact,
            latencyMs: imagePayload.latency_ms,
            detections: imagePayload.detections.length
          },
          ...current
        ].slice(0, 6));
      } else {
        const nextVideoResponse = payload as VideoResponse;
        setVideoResponse(nextVideoResponse);
        setRunHistory((current) => [
          {
            label: `${nextVideoResponse.model_id} · ${nextVideoResponse.runtime} video`,
            artifact: nextVideoResponse.artifact,
            latencyMs: nextVideoResponse.average_latency_ms,
            detections: nextVideoResponse.frame_results.reduce(
              (sum, frame) => sum + frame.detections.length,
              0
            )
          },
          ...current
        ].slice(0, 6));
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unexpected request error."
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">OD</div>
          <div>
            <p className="brand-title">Object Detection Studio</p>
            <p className="brand-subtitle">Inference benchmarking workspace</p>
          </div>
        </div>
        <div className="topbar-meta">
          <div className="status-pill">
            <span className="status-dot" />
            {availabilityError ? "Backend check required" : "Artifacts connected"}
          </div>
          <div className="topbar-chip">{selectedArtifact}</div>
        </div>
      </header>

      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Professional Inference Dashboard</p>
          <h1>Run, compare, and review object detection results in one place.</h1>
          <p className="lede">
            Use your own images or videos to compare YOLO runtimes, inspect detections, and collect latency numbers for your assignment report.
          </p>
          <div className="hero-tags">
            <span>FastAPI backend</span>
            <span>YOLO runtime switching</span>
            <span>Upload-based testing</span>
          </div>
        </div>

        <div className="hero-summary">
          <div className="summary-card accent-card">
            <p className="meta-label">Current Setup</p>
            <p className="summary-value">{modelId}</p>
            <p className="summary-footnote">{runtime} runtime</p>
          </div>
          <div className="summary-card">
            <p className="meta-label">Latency</p>
            <p className="summary-value">{activeLatency ? `${activeLatency.toFixed(2)} ms` : "--"}</p>
            <p className="summary-footnote">Latest run</p>
          </div>
          <div className="summary-card">
            <p className="meta-label">Detections</p>
            <p className="summary-value">{activeDetections}</p>
            <p className="summary-footnote">{activeMode} mode</p>
          </div>
          <div className="summary-card">
            <p className="meta-label">Coverage</p>
            <p className="summary-value">
              {modelOptions.length} x {runtimeOptions.length}
            </p>
            <p className="summary-footnote">Models and runtimes</p>
          </div>
        </div>
      </section>

      <section className="workspace">
        <form className="control-panel" onSubmit={onSubmit}>
          <div className="panel-heading">
            <div>
              <p className="panel-eyebrow">Control Panel</p>
              <h2>Inference Inputs</h2>
            </div>
            <span className="panel-badge">{activeMode}</span>
          </div>

          <label className="field">
            <span>Upload media</span>
            <input type="file" accept="image/*,video/*" onChange={onFileChange} />
          </label>

          <label className="field">
            <span>Model</span>
            <select value={modelId} onChange={(event) => setModelId(event.target.value)}>
              {modelOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Runtime</span>
            <select value={runtime} onChange={(event) => setRuntime(event.target.value)}>
              {runtimeOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>Confidence Threshold</span>
            <input
              max="0.95"
              min="0.05"
              onChange={(event) => setConfidence(event.target.value)}
              step="0.05"
              type="number"
              value={confidence}
            />
          </label>

          <label className="field">
            <span>IoU Threshold</span>
            <input
              max="0.95"
              min="0.05"
              onChange={(event) => setIou(event.target.value)}
              step="0.05"
              type="number"
              value={iou}
            />
          </label>

          <label className="field">
            <span>Image Size</span>
            <select value={imgsz} onChange={(event) => setImgsz(event.target.value)}>
              {["416", "640", "960", "1280"].map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          <button className="submit-button" type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Running..." : "Run Detection"}
          </button>

          <div className="artifact-panel">
            <p className="meta-label">Artifact In Use</p>
            <p className="meta-value">{selectedArtifact}</p>
          </div>

          <div className="sidebar-metrics">
            <div className="mini-metric">
              <span>Models</span>
              <strong>{modelOptions.length}</strong>
            </div>
            <div className="mini-metric">
              <span>Runtimes</span>
              <strong>{runtimeOptions.length}</strong>
            </div>
            <div className="mini-metric">
              <span>History</span>
              <strong>{runHistory.length}</strong>
            </div>
          </div>

          {availabilityError ? <p className="error-text">{availabilityError}</p> : null}
          {error ? <p className="error-text">{error}</p> : null}
        </form>

        <div className="content-grid">
          {previewUrl && isImage ? (
            <section className="stage-card">
              <div className="card-header">
                <div>
                  <p className="panel-eyebrow">Preview</p>
                  <h2>{previewLabel}</h2>
                </div>
                <span className="panel-badge">Annotated overlay</span>
              </div>

              <div className="image-stage">
                <div className="image-canvas">
                  <img alt="Upload preview" className="preview-image" src={previewUrl} />
                  {response?.detections.map((detection, index) => {
                    const [x1, y1, x2, y2] = detection.bbox_xyxy;
                    const width = response.image_width;
                    const height = response.image_height;
                    return (
                      <div
                        className="box"
                        key={`${detection.class_name}-${index}`}
                        style={{
                          left: `${clamp((x1 / width) * 100, 0, 100)}%`,
                          top: `${clamp((y1 / height) * 100, 0, 100)}%`,
                          width: `${clamp(((x2 - x1) / width) * 100, 0, 100)}%`,
                          height: `${clamp(((y2 - y1) / height) * 100, 0, 100)}%`
                        }}
                      >
                        <span>
                          {detection.class_name} {(detection.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>
          ) : (
            <section className="empty-state stage-card">
              {videoResponse ? (
                <div className="video-stage">
                  <div className="card-header">
                    <div>
                      <p className="panel-eyebrow">Video Analytics</p>
                      <h2>{previewLabel}</h2>
                    </div>
                    <span className="panel-badge">{videoResponse.frames_processed} sampled frames</span>
                  </div>

                  <div className="video-metric-grid">
                    <div className="result-tile">
                      <span>Total detections</span>
                      <strong>{totalVideoDetections}</strong>
                    </div>
                    <div className="result-tile">
                      <span>Avg detections/frame</span>
                      <strong>{averageDetectionsPerFrame.toFixed(2)}</strong>
                    </div>
                    <div className="result-tile">
                      <span>Avg latency</span>
                      <strong>{videoResponse.average_latency_ms.toFixed(2)} ms</strong>
                    </div>
                    <div className="result-tile">
                      <span>Video duration</span>
                      <strong>{(videoResponse.total_video_ms / 1000).toFixed(2)} s</strong>
                    </div>
                  </div>

                  <div className="video-breakdown-grid">
                    <div className="results-card inset-card">
                      <p className="list-title">Top classes across video</p>
                      {videoClassBreakdown.length > 0 ? (
                        videoClassBreakdown.map((item) => (
                          <div className="detail-row" key={item.className}>
                            <span>{item.className}</span>
                            <strong>{item.count}</strong>
                          </div>
                        ))
                      ) : (
                        <div className="no-results">
                          <p>No detections found.</p>
                        </div>
                      )}
                    </div>

                    <div className="results-card inset-card">
                      <p className="list-title">Per-frame summary</p>
                      <div className="frame-table">
                        {videoResponse.frame_results.slice(0, 8).map((frame) => (
                          <div className="frame-row" key={frame.frame_index}>
                            <div>
                              <p className="history-title">Frame {frame.frame_index}</p>
                              <p className="history-artifact">{frame.timestamp_ms.toFixed(0)} ms</p>
                            </div>
                            <div className="history-stats">
                              <strong>{frame.detections.length} detections</strong>
                              <span>{frame.latency_ms.toFixed(2)} ms</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="empty-illustration">
                    <div className="empty-orbit empty-orbit-one" />
                    <div className="empty-orbit empty-orbit-two" />
                    <div className="empty-core">+</div>
                  </div>
                  <h2>Drop in media to start a run</h2>
                  <p>
                    Upload an image for bounding-box visualization or a video for frame-based latency analysis.
                  </p>
                </>
              )}
            </section>
          )}

          <section className="results-card primary-results">
            <div className="card-header">
              <div>
                <p className="panel-eyebrow">Output</p>
                <h2>Run Summary</h2>
              </div>
              <span className="panel-badge">
                {response ? "Image result" : videoResponse ? "Video result" : "Awaiting run"}
              </span>
            </div>

            {response ? (
              <>
                <div className="result-grid">
                  <div className="result-tile">
                    <span>Model</span>
                    <strong>{response.model_id}</strong>
                  </div>
                  <div className="result-tile">
                    <span>Runtime</span>
                    <strong>{response.runtime}</strong>
                  </div>
                  <div className="result-tile">
                    <span>Latency</span>
                    <strong>{response.latency_ms.toFixed(2)} ms</strong>
                  </div>
                  <div className="result-tile">
                    <span>Detections</span>
                    <strong>{response.detections.length}</strong>
                  </div>
                </div>

                <div className="detail-list">
                  <div className="detail-row">
                    <span>Artifact</span>
                    <strong>{response.artifact}</strong>
                  </div>
                  <div className="detail-row">
                    <span>Resolution</span>
                    <strong>
                      {response.image_width} x {response.image_height}
                    </strong>
                  </div>
                </div>

                <div className="detection-list">
                  <p className="list-title">Top detections</p>
                  {response.detections.slice(0, 6).map((detection, index) => (
                    <div className="detail-row" key={`${detection.class_name}-${index}`}>
                      <span>{detection.class_name}</span>
                      <strong>{(detection.confidence * 100).toFixed(1)}%</strong>
                    </div>
                  ))}
                </div>
              </>
            ) : videoResponse ? (
              <>
                <div className="result-grid">
                  <div className="result-tile">
                    <span>Model</span>
                    <strong>{videoResponse.model_id}</strong>
                  </div>
                  <div className="result-tile">
                    <span>Runtime</span>
                    <strong>{videoResponse.runtime}</strong>
                  </div>
                  <div className="result-tile">
                    <span>Avg latency</span>
                    <strong>{videoResponse.average_latency_ms.toFixed(2)} ms</strong>
                  </div>
                  <div className="result-tile">
                    <span>Frames</span>
                    <strong>{videoResponse.frames_processed}</strong>
                  </div>
                  <div className="result-tile">
                    <span>Total detections</span>
                    <strong>{totalVideoDetections}</strong>
                  </div>
                  <div className="result-tile">
                    <span>Detections / frame</span>
                    <strong>{averageDetectionsPerFrame.toFixed(2)}</strong>
                  </div>
                </div>

                <div className="detail-list">
                  <div className="detail-row">
                    <span>Artifact</span>
                    <strong>{videoResponse.artifact}</strong>
                  </div>
                  <div className="detail-row">
                    <span>Video duration</span>
                    <strong>{videoResponse.total_video_ms.toFixed(2)} ms</strong>
                  </div>
                </div>

                <div className="detection-list">
                  <p className="list-title">Top classes across sampled frames</p>
                  {videoClassBreakdown.map((item) => (
                    <div className="detail-row" key={item.className}>
                      <span>{item.className}</span>
                      <strong>{item.count}</strong>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="no-results">
                <p>No inference result yet.</p>
                <span>Choose a file, select a runtime, and launch a run.</span>
              </div>
            )}
          </section>

          <section className="results-card history-card">
            <div className="card-header">
              <div>
                <p className="panel-eyebrow">Comparison Log</p>
                <h2>Recent Runs</h2>
              </div>
              <span className="panel-badge">{runHistory.length} stored</span>
            </div>

            {runHistory.length > 0 ? (
              <div className="history-table">
                {runHistory.map((run, index) => (
                  <div className="history-row" key={`${run.label}-${index}`}>
                    <div>
                      <p className="history-title">{run.label}</p>
                      <p className="history-artifact">{run.artifact}</p>
                    </div>
                    <div className="history-stats">
                      <strong>{run.latencyMs.toFixed(2)} ms</strong>
                      <span>{run.detections} detections</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-results">
                <p>No comparison history yet.</p>
                <span>Completed runs will appear here for quick review.</span>
              </div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}
