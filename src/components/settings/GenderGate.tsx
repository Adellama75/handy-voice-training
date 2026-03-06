import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { Slider } from "../ui/Slider";
import { useSettings } from "../../hooks/useSettings";
import { commands } from "@/bindings";
import { listen } from "@tauri-apps/api/event";

interface GenderGateProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const GenderGate: React.FC<GenderGateProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { settings, getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("gender_gate_enabled") || false;
    const threshold = settings?.gender_gate_threshold ?? 0.5;

    const [isDownloaded, setIsDownloaded] = useState<boolean | null>(null);
    const [isDownloading, setIsDownloading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [downloadError, setDownloadError] = useState<string | null>(null);

    useEffect(() => {
      commands.getGenderModelStatus().then((res) => {
        if (res.status === "ok") setIsDownloaded(res.data.is_downloaded);
      });

      const unlistenProgress = listen<{ percentage: number }>(
        "gender-model-progress",
        (e) => setProgress(e.payload.percentage),
      );
      const unlistenReady = listen("gender-model-ready", () => {
        setIsDownloaded(true);
        setIsDownloading(false);
        setProgress(0);
      });

      return () => {
        unlistenProgress.then((f) => f());
        unlistenReady.then((f) => f());
      };
    }, []);

    const handleDownload = async () => {
      setIsDownloading(true);
      setDownloadError(null);
      setProgress(0);
      const res = await commands.downloadGenderModel();
      if (res.status === "error") {
        setDownloadError(res.error);
        setIsDownloading(false);
      }
    };

    return (
      <>
        <ToggleSwitch
          checked={enabled}
          onChange={(value) => updateSetting("gender_gate_enabled", value)}
          isUpdating={isUpdating("gender_gate_enabled")}
          label={t("settings.genderGate.toggle.label")}
          description={t("settings.genderGate.toggle.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        />

        {isDownloaded === false && (
          <div className="px-4 py-3 text-sm text-text/80 flex flex-col gap-2">
            <span>{t("settings.genderGate.model.notDownloaded")}</span>
            {isDownloading ? (
              <div className="flex flex-col gap-1">
                <div className="w-full bg-mid-gray/20 rounded-full h-2">
                  <div
                    className="bg-background-ui h-2 rounded-full transition-all"
                    style={{ width: `${progress.toFixed(1)}%` }}
                  />
                </div>
                <span className="text-xs text-text/50">
                  {progress.toFixed(1)}%
                </span>
              </div>
            ) : (
              <button
                onClick={handleDownload}
                className="self-start px-3 py-1.5 rounded-lg bg-background-ui text-white text-xs font-medium hover:opacity-90 transition-opacity"
              >
                {t("settings.genderGate.model.download")}
              </button>
            )}
            {downloadError && (
              <span className="text-xs text-red-400">{downloadError}</span>
            )}
          </div>
        )}

        {enabled && isDownloaded && (
          <Slider
            value={threshold}
            onChange={(value) => updateSetting("gender_gate_threshold", value)}
            min={0.0}
            max={1.0}
            label={t("settings.genderGate.threshold.label")}
            description={t("settings.genderGate.threshold.description")}
            descriptionMode={descriptionMode}
            grouped={grouped}
          />
        )}
      </>
    );
  },
);
