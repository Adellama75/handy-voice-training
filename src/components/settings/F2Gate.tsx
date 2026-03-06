import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { Slider } from "../ui/Slider";
import { useSettings } from "../../hooks/useSettings";

interface F2GateProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const F2Gate: React.FC<F2GateProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { settings, getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("f2_gate_enabled") || false;
    const minHz = settings?.f2_gate_min_hz ?? 1300;

    return (
      <>
        <ToggleSwitch
          checked={enabled}
          onChange={(value) => updateSetting("f2_gate_enabled", value)}
          isUpdating={isUpdating("f2_gate_enabled")}
          label={t("settings.f2Gate.toggle.label")}
          description={t("settings.f2Gate.toggle.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        />
        {enabled && (
          <Slider
            value={minHz}
            onChange={(value) => updateSetting("f2_gate_min_hz", value)}
            min={800}
            max={2000}
            label={t("settings.f2Gate.minHz.label")}
            description={t("settings.f2Gate.minHz.description")}
            descriptionMode={descriptionMode}
            grouped={grouped}
          />
        )}
      </>
    );
  },
);
