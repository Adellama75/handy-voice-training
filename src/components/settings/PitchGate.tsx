import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { Slider } from "../ui/Slider";
import { useSettings } from "../../hooks/useSettings";

interface PitchGateProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const PitchGate: React.FC<PitchGateProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { settings, getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("pitch_gate_enabled") || false;
    const minHz = settings?.pitch_gate_min_hz ?? 165;

    return (
      <>
        <ToggleSwitch
          checked={enabled}
          onChange={(value) => updateSetting("pitch_gate_enabled", value)}
          isUpdating={isUpdating("pitch_gate_enabled")}
          label={t("settings.pitchGate.toggle.label")}
          description={t("settings.pitchGate.toggle.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        />
        {enabled && (
          <Slider
            value={minHz}
            onChange={(value) => updateSetting("pitch_gate_min_hz", value)}
            min={50}
            max={300}
            label={t("settings.pitchGate.minHz.label")}
            description={t("settings.pitchGate.minHz.description")}
            descriptionMode={descriptionMode}
            grouped={grouped}
          />
        )}
      </>
    );
  },
);
