"use client";

import * as React from "react";
import { Sparkles, Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface GenerateSummaryButtonProps {
  onClick: () => void;
  isLoading?: boolean;
  disabled?: boolean;
  className?: string;
}

export function GenerateSummaryButton({
  onClick,
  isLoading = false,
  disabled = false,
  className,
}: GenerateSummaryButtonProps) {
  const [isHovering, setIsHovering] = React.useState(false);

  return (
    <motion.button
      onClick={onClick}
      disabled={disabled || isLoading}
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        "group relative rounded-full bg-gradient-to-r from-blue-300/30 via-blue-500/30 via-40% to-purple-500/30 p-1 transition-all",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        className
      )}
    >
      <div className="relative flex items-center justify-center gap-2 rounded-full bg-gradient-to-r from-blue-400 via-blue-600 via-40% to-purple-600 px-4 py-2 text-white overflow-hidden">
        {/* Sparkle decorations */}
        <Sparkles
          className={cn(
            "size-5 transition-all duration-300",
            isHovering ? "animate-sparkle" : ""
          )}
        />
        <Sparkles
          className={cn(
            "absolute bottom-2 left-3 z-20 size-2 rotate-12 opacity-70",
            isHovering ? "animate-sparkle-delayed" : "opacity-0"
          )}
          style={{ animationDelay: "0.3s" }}
        />
        <Sparkles
          className={cn(
            "absolute right-4 top-2 size-1.5 -rotate-12 opacity-70",
            isHovering ? "animate-sparkle-delayed" : "opacity-0"
          )}
          style={{ animationDelay: "0.6s" }}
        />

        {/* Shimmer effect on hover */}
        {isHovering && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
            initial={{ x: "-100%" }}
            animate={{ x: "100%" }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
          />
        )}

        <span className="font-semibold text-sm whitespace-nowrap">
          {isLoading ? (
            <span className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Generating...
            </span>
          ) : (
            "Generate Summary"
          )}
        </span>
      </div>
    </motion.button>
  );
}
