import { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, Check } from "lucide-react";

export type LanguageCode = "en" | "fr";

export interface Language {
  code: LanguageCode;
  label: string;
  flag: string;
  triageSystem: "english" | "french"; // Maps to triage system
}

// Only English and French for medical triage
const languages: Language[] = [
  { code: "en", label: "English", flag: "🇬🇧", triageSystem: "english" },
  { code: "fr", label: "Français", flag: "🇫🇷", triageSystem: "french" },
];

interface LanguageSelectorDropdownProps {
  value?: LanguageCode;
  onChange?: (language: Language) => void;
  disabled?: boolean;
}

export const LanguageSelectorDropdown = ({
  value = "en",
  onChange,
  disabled = false,
}: LanguageSelectorDropdownProps) => {
  const [selected, setSelected] = useState(
    languages.find((l) => l.code === value) || languages[0]
  );
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Update selected when value prop changes
  useEffect(() => {
    const lang = languages.find((l) => l.code === value);
    if (lang) setSelected(lang);
  }, [value]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (lang: Language) => {
    setSelected(lang);
    setOpen(false);
    onChange?.(lang);
  };

  return (
    <div className="relative inline-block" ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        onClick={() => !disabled && setOpen((o) => !o)}
        disabled={disabled}
        className={cn(
          "flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm",
          "bg-white/60 dark:bg-neutral-900/90 backdrop-blur-md shadow-sm",
          "border-gray-200 dark:border-neutral-700",
          "text-gray-800 dark:text-neutral-200",
          "hover:bg-gray-50 dark:hover:bg-neutral-800 transition-all",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <span>{selected.flag}</span>
        <span>{selected.label}</span>
        <ChevronDown className="h-4 w-4" />
      </button>

      {/* Dropdown Menu */}
      {open && !disabled && (
        <div
          className={cn(
            "absolute left-0 mt-2 w-48 rounded-xl overflow-hidden z-50",
            "bg-white/90 dark:bg-neutral-900/95 backdrop-blur-xl",
            "shadow-lg border border-gray-200 dark:border-neutral-700",
            "animate-in fade-in-0 zoom-in-95 duration-200"
          )}
        >
          {languages.map((lang) => (
            <button
              key={lang.code}
              onClick={() => handleSelect(lang)}
              className={cn(
                "flex items-center gap-2 w-full px-3 py-2 text-sm text-left transition-colors",
                selected.code === lang.code
                  ? "font-semibold text-blue-600 dark:text-blue-400"
                  : "text-gray-800 dark:text-neutral-200 hover:bg-gray-100 dark:hover:bg-neutral-800"
              )}
            >
              <span>{lang.flag}</span>
              <span className="flex-1">{lang.label}</span>
              {selected.code === lang.code && (
                <Check className="h-4 w-4 text-blue-500 dark:text-blue-400" />
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export { languages };
export default LanguageSelectorDropdown;
