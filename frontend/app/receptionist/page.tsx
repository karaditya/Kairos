"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Mic, MicOff, Upload, Play, Pause, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { caeApi } from "@/lib/api";
import { t, type LanguageCode } from "@/lib/translations";
import { useAudioSession } from "@/hooks/useWebSocket";
import type { SessionStartResponse, TranscriptSegment } from "@/lib/types";

export default function ReceptionistDashboard() {
  const searchParams = useSearchParams();
  const lang = (searchParams.get("lang") || "en") as LanguageCode;

  // Session state
  const [session, setSession] = useState<SessionStartResponse | null>(null);
  const [sessionStatus, setSessionStatus] = useState<"idle" | "active" | "paused" | "completed">("idle");
  const [patientId, setPatientId] = useState("");
  const [sessionLanguage, setSessionLanguage] = useState<"fr" | "en">(lang === "fr" ? "fr" : "en");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Audio session hook
  const {
    isConnected,
    isConnecting,
    isRecording,
    isSupported,
    transcript,
    wsError,
    recordingError,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    clearTranscript,
  } = useAudioSession({
    sessionId: session?.session_id || null,
    language: sessionLanguage,
  });

  // File upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Start a new session
  const handleStartSession = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await caeApi.session.start({
        patient_id: patientId || undefined,
        language: sessionLanguage,
      });
      setSession(response);
      setSessionStatus("active");
      clearTranscript();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorStartSession", lang));
    } finally {
      setIsLoading(false);
    }
  };

  // Pause session
  const handlePauseSession = async () => {
    if (!session) return;
    try {
      await caeApi.session.pause(session.session_id);
      setSessionStatus("paused");
      stopRecording();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to pause session");
    }
  };

  // Resume session
  const handleResumeSession = async () => {
    if (!session) return;
    try {
      await caeApi.session.resume(session.session_id);
      setSessionStatus("active");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to resume session");
    }
  };

  // End session
  const handleEndSession = async () => {
    if (!session) return;
    try {
      stopRecording();
      disconnect();
      await caeApi.session.end(session.session_id);
      setSessionStatus("completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorEndSession", lang));
    }
  };

  // Toggle recording
  const handleToggleRecording = async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  };

  // Upload audio file
  const handleUploadAudio = async () => {
    if (!session || !uploadFile) return;
    setIsUploading(true);
    setError(null);
    try {
      await caeApi.audio.upload(session.session_id, uploadFile, sessionLanguage);
      // Refresh transcript
      const transcriptData = await caeApi.audio.getTranscript(session.session_id);
      // Note: transcript is managed by the hook, but we could update it here if needed
      setUploadFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorUploadAudio", lang));
    } finally {
      setIsUploading(false);
    }
  };

  // Reset for new session
  const handleNewSession = () => {
    setSession(null);
    setSessionStatus("idle");
    setPatientId("");
    clearTranscript();
    setError(null);
  };

  // Format timestamp
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Header */}
      <header className="border-b bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href={`/?lang=${lang}`}>
              <Button variant="ghost" size="sm">
                <ArrowLeft className="h-4 w-4 mr-2" />
                {t("backToHome", lang)}
              </Button>
            </Link>
            <h1 className="text-2xl font-extralight tracking-wider text-indigo-900 dark:text-white">
              {t("receptionistDashboard", lang)}
            </h1>
          </div>
          <div className="flex items-center gap-2">
            {session && (
              <span className={`px-3 py-1 rounded-full text-sm ${
                sessionStatus === "active" ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" :
                sessionStatus === "paused" ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200" :
                sessionStatus === "completed" ? "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200" :
                "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
              }`}>
                {sessionStatus === "active" ? t("sessionActive", lang) :
                 sessionStatus === "paused" ? t("sessionPaused", lang) :
                 sessionStatus === "completed" ? t("sessionEnded", lang) : "Idle"}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
            {error}
          </div>
        )}

        {sessionStatus === "idle" ? (
          /* Session Start Form */
          <Card className="max-w-md mx-auto">
            <CardHeader>
              <CardTitle className="font-light tracking-wide">{t("newSession", lang)}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>{t("patientId", lang)} ({t("optional", lang) || "Optional"})</Label>
                <Input
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="P-12345"
                />
              </div>
              <div>
                <Label>{t("language", lang)}</Label>
                <div className="flex gap-2 mt-2">
                  <Button
                    variant={sessionLanguage === "fr" ? "default" : "outline"}
                    onClick={() => setSessionLanguage("fr")}
                    className="flex-1"
                  >
                    {t("french", lang)}
                  </Button>
                  <Button
                    variant={sessionLanguage === "en" ? "default" : "outline"}
                    onClick={() => setSessionLanguage("en")}
                    className="flex-1"
                  >
                    {t("english", lang)}
                  </Button>
                </div>
              </div>
              <Button
                onClick={handleStartSession}
                disabled={isLoading}
                className="w-full bg-indigo-600 hover:bg-indigo-700"
              >
                {isLoading ? t("processing", lang) : t("startSession", lang)}
              </Button>
            </CardContent>
          </Card>
        ) : (
          /* Active Session View */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Controls */}
            <div className="space-y-6">
              {/* Session Info */}
              <Card>
                <CardHeader>
                  <CardTitle className="font-light text-sm">{t("sessionDetails", lang)}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">{t("sessionId", lang)}:</span>
                    <span className="font-mono">{session?.session_id.slice(0, 8)}...</span>
                  </div>
                  {patientId && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">{t("patientId", lang)}:</span>
                      <span>{patientId}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-gray-500">{t("language", lang)}:</span>
                    <span>{sessionLanguage === "fr" ? t("french", lang) : t("english", lang)}</span>
                  </div>
                </CardContent>
              </Card>

              {/* Session Controls */}
              <Card>
                <CardHeader>
                  <CardTitle className="font-light text-sm">Controls</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {sessionStatus !== "completed" && (
                    <>
                      <div className="flex gap-2">
                        {sessionStatus === "active" ? (
                          <Button
                            onClick={handlePauseSession}
                            variant="outline"
                            className="flex-1"
                          >
                            <Pause className="h-4 w-4 mr-2" />
                            {t("pauseSession", lang)}
                          </Button>
                        ) : (
                          <Button
                            onClick={handleResumeSession}
                            variant="outline"
                            className="flex-1"
                          >
                            <Play className="h-4 w-4 mr-2" />
                            {t("resumeSession", lang)}
                          </Button>
                        )}
                      </div>
                      <Button
                        onClick={handleEndSession}
                        variant="destructive"
                        className="w-full"
                      >
                        <Square className="h-4 w-4 mr-2" />
                        {t("endSession", lang)}
                      </Button>
                    </>
                  )}
                  {sessionStatus === "completed" && (
                    <Button onClick={handleNewSession} className="w-full">
                      {t("newSession", lang)}
                    </Button>
                  )}
                </CardContent>
              </Card>

              {/* Audio Controls */}
              {sessionStatus === "active" && (
                <Card>
                  <CardHeader>
                    <CardTitle className="font-light text-sm">{t("audioRecording", lang)}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {isSupported ? (
                      <>
                        <Button
                          onClick={handleToggleRecording}
                          className={`w-full ${isRecording ? "bg-red-600 hover:bg-red-700" : "bg-indigo-600 hover:bg-indigo-700"}`}
                          disabled={isConnecting}
                        >
                          {isRecording ? (
                            <>
                              <MicOff className="h-4 w-4 mr-2" />
                              {t("stopRecording", lang)}
                            </>
                          ) : (
                            <>
                              <Mic className="h-4 w-4 mr-2" />
                              {t("startRecording", lang)}
                            </>
                          )}
                        </Button>
                        {isRecording && (
                          <div className="flex items-center justify-center gap-2 text-sm text-red-600">
                            <span className="w-2 h-2 bg-red-600 rounded-full animate-pulse" />
                            Recording...
                          </div>
                        )}
                        {(wsError || recordingError) && (
                          <p className="text-sm text-red-500">{wsError || recordingError}</p>
                        )}
                      </>
                    ) : (
                      <p className="text-sm text-gray-500">Audio recording not supported in this browser.</p>
                    )}

                    <div className="border-t pt-4">
                      <Label className="text-sm">{t("uploadAudio", lang)}</Label>
                      <Input
                        type="file"
                        accept="audio/*,.wav,.mp3,.m4a"
                        onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                        className="mt-2"
                      />
                      {uploadFile && (
                        <Button
                          onClick={handleUploadAudio}
                          disabled={isUploading}
                          className="w-full mt-2"
                          variant="outline"
                        >
                          <Upload className="h-4 w-4 mr-2" />
                          {isUploading ? t("processing", lang) : t("uploadAudio", lang)}
                        </Button>
                      )}
                      <p className="text-xs text-gray-500 mt-2">{t("supportedFormats", lang)}</p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Right Column - Transcript */}
            <div className="lg:col-span-2">
              <Card className="h-full">
                <CardHeader>
                  <CardTitle className="font-light flex items-center gap-2">
                    {t("liveTranscript", lang)}
                    {isConnected && (
                      <span className="w-2 h-2 bg-green-500 rounded-full" title="Connected" />
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {transcript.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                      <Mic className="h-12 w-12 mx-auto mb-4 opacity-30" />
                      <p>{t("transcriptEmpty", lang)}</p>
                    </div>
                  ) : (
                    <div className="space-y-3 max-h-[600px] overflow-y-auto">
                      {transcript.map((segment, index) => (
                        <div key={index} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                          <div className="flex items-start gap-3">
                            <span className="text-xs text-gray-400 font-mono whitespace-nowrap">
                              {formatTime(segment.start_time)}
                            </span>
                            <div className="flex-1">
                              <p className="text-sm">{segment.text}</p>
                              {segment.keywords.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-2">
                                  {segment.keywords.map((kw, i) => (
                                    <span
                                      key={i}
                                      className="px-2 py-0.5 bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 text-xs rounded-full"
                                    >
                                      {kw}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                            <span className="text-xs text-gray-400">
                              {Math.round(segment.confidence * 100)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
