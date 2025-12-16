"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface AnimatedDownloadButtonProps {
  onClick: () => void;
  filename?: string;
  className?: string;
  onExpand?: () => void;
}

export function AnimatedDownloadButton({
  onClick,
  filename = "Download",
  className,
  onExpand,
}: AnimatedDownloadButtonProps) {
  const [isHovered, setIsHovered] = React.useState(false);
  const [isClicked, setIsClicked] = React.useState(false);

  const handleClick = () => {
    setIsClicked(true);
    onClick();
    setTimeout(() => setIsClicked(false), 2000);
  };

  const handleHoverStart = () => {
    setIsHovered(true);
    onExpand?.();
  };

  return (
    <motion.button
      onClick={handleClick}
      initial={{ width: 48, height: 48, opacity: 0, scale: 0.8 }}
      animate={{
        width: isHovered ? 180 : 48,
        height: 48,
        opacity: 1,
        scale: 1,
      }}
      onHoverStart={handleHoverStart}
      onHoverEnd={() => setIsHovered(false)}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={cn(
        "bg-gradient-to-r from-green-500 to-emerald-600 flex items-center justify-center overflow-hidden relative shadow-lg hover:shadow-xl",
        "focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-offset-2",
        className
      )}
      style={{ borderRadius: 24 }}
    >
      {/* Icon state */}
      <AnimatePresence mode="wait">
        {!isHovered && !isClicked && (
          <motion.div
            key="icon"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2 }}
            className="absolute"
          >
            <Download className="h-5 w-5 text-white" />
          </motion.div>
        )}

        {isHovered && !isClicked && (
          <motion.div
            key="text"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2, delay: 0.1 }}
            className="w-full flex justify-center items-center gap-2"
          >
            <Download className="h-4 w-4 text-white" />
            <span className="text-white text-sm font-semibold whitespace-nowrap">
              Download PDF
            </span>
          </motion.div>
        )}

        {isClicked && (
          <motion.div
            key="success"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-2"
          >
            <Check className="h-5 w-5 text-white" />
            {isHovered && (
              <span className="text-white text-sm font-semibold">Done!</span>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Shine effect */}
      {isHovered && (
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
          initial={{ x: "-100%" }}
          animate={{ x: "100%" }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
        />
      )}
    </motion.button>
  );
}
