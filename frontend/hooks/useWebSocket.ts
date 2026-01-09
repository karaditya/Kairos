// =============================================================================
// WebSocket Hook for Audio Streaming
// =============================================================================

import { useState, useEffect, useRef, useCallback } from "react";
import { caeApi } from "@/lib/api";
import type { TranscriptSegment, WSMessage } from "@/lib/types";

export interface UseAudioWebSocketOptions {
  sessionId: string | null;
  onTranscript?: (segment: TranscriptSegment) => void;
  onStatus?: (status: string, message?: string) => void;
  onError?: (error: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export interface UseAudioWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  transcript: TranscriptSegment[];
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  sendAudio: (data: ArrayBuffer) => void;
  clearTranscript: () => void;
}

export function useAudioWebSocket(
  options: UseAudioWebSocketOptions
): UseAudioWebSocketReturn {
  const {
    sessionId,
    onTranscript,
    onStatus,
    onError,
    onConnect,
    onDisconnect,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const clearTranscript = useCallback(() => {
    setTranscript([]);
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  const connect = useCallback(() => {
    if (!sessionId) {
      setError("No session ID provided");
      return;
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    setIsConnecting(true);
    setError(null);

    const wsUrl = caeApi.audio.getWebSocketUrl(sessionId);
    const ws = new WebSocket(wsUrl);

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setIsConnected(true);
      setIsConnecting(false);
      setError(null);
      onConnect?.();
    };

    ws.onmessage = (event) => {
      try {
        // Handle text messages (JSON)
        if (typeof event.data === "string") {
          const message: WSMessage = JSON.parse(event.data);

          switch (message.type) {
            case "transcript":
              setTranscript((prev) => [...prev, message.segment]);
              onTranscript?.(message.segment);
              break;

            case "status":
              onStatus?.(message.status, message.message);
              break;

            case "error":
              setError(message.error);
              onError?.(message.error);
              break;
          }
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onerror = (event) => {
      console.error("WebSocket error:", event);
      setError("WebSocket connection error");
      onError?.("WebSocket connection error");
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      setIsConnecting(false);
      onDisconnect?.();

      // Don't reconnect if closed normally
      if (event.code !== 1000 && event.code !== 1001) {
        setError(`Connection closed: ${event.reason || "Unknown reason"}`);
      }
    };

    wsRef.current = ws;
  }, [sessionId, onTranscript, onStatus, onError, onConnect, onDisconnect]);

  const sendAudio = useCallback((data: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Disconnect when session changes
  useEffect(() => {
    if (!sessionId) {
      disconnect();
    }
  }, [sessionId, disconnect]);

  return {
    isConnected,
    isConnecting,
    transcript,
    error,
    connect,
    disconnect,
    sendAudio,
    clearTranscript,
  };
}

// =============================================================================
// Audio Recording Hook
// =============================================================================

export interface UseAudioRecorderOptions {
  onAudioData?: (data: ArrayBuffer) => void;
  sampleRate?: number;
}

export interface UseAudioRecorderReturn {
  isRecording: boolean;
  isSupported: boolean;
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
}

export function useAudioRecorder(
  options: UseAudioRecorderOptions = {}
): UseAudioRecorderReturn {
  const { onAudioData, sampleRate = 16000 } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const isSupported =
    typeof window !== "undefined" &&
    !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  const stopRecording = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }

    setIsRecording(false);
  }, []);

  const startRecording = useCallback(async () => {
    if (!isSupported) {
      setError("Audio recording is not supported in this browser");
      return;
    }

    try {
      setError(null);

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      streamRef.current = stream;

      // Create audio context for processing
      const audioContext = new AudioContext({ sampleRate });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);

      // Create script processor for raw audio data
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (onAudioData) {
          const inputData = e.inputBuffer.getChannelData(0);
          // Convert Float32Array to Int16Array for transmission
          const int16Data = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          onAudioData(int16Data.buffer);
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to start recording";
      setError(message);
      console.error("Recording error:", err);
    }
  }, [isSupported, sampleRate, onAudioData]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return {
    isRecording,
    isSupported,
    error,
    startRecording,
    stopRecording,
  };
}

// =============================================================================
// Combined Audio Session Hook
// =============================================================================

export interface UseAudioSessionOptions {
  sessionId: string | null;
  language?: string;
}

export interface UseAudioSessionReturn {
  // WebSocket state
  isConnected: boolean;
  isConnecting: boolean;
  wsError: string | null;

  // Recording state
  isRecording: boolean;
  recordingError: string | null;
  isSupported: boolean;

  // Transcript
  transcript: TranscriptSegment[];

  // Actions
  connect: () => void;
  disconnect: () => void;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  clearTranscript: () => void;
}

export function useAudioSession(
  options: UseAudioSessionOptions
): UseAudioSessionReturn {
  const { sessionId } = options;

  const {
    isConnected,
    isConnecting,
    transcript,
    error: wsError,
    connect,
    disconnect,
    sendAudio,
    clearTranscript,
  } = useAudioWebSocket({ sessionId });

  const {
    isRecording,
    isSupported,
    error: recordingError,
    startRecording: startRec,
    stopRecording,
  } = useAudioRecorder({
    onAudioData: sendAudio,
  });

  const startRecording = useCallback(async () => {
    // Ensure WebSocket is connected first
    if (!isConnected) {
      connect();
      // Wait a bit for connection
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    await startRec();
  }, [isConnected, connect, startRec]);

  return {
    isConnected,
    isConnecting,
    wsError,
    isRecording,
    recordingError,
    isSupported,
    transcript,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    clearTranscript,
  };
}
