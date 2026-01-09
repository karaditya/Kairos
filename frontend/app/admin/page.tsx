"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Settings,
  Database,
  Search,
  Plus,
  Trash2,
  RefreshCw,
  Download,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { caeApi } from "@/lib/api";
import { t, type LanguageCode } from "@/lib/translations";
import type {
  ProtocolStats,
  ProtocolSearchResult,
  OllamaModel,
  SystemStatus,
  PullProgress,
} from "@/lib/types";

export default function AdminDashboard() {
  const searchParams = useSearchParams();
  const lang = (searchParams.get("lang") || "en") as LanguageCode;

  // Auth state
  const [staffPin, setStaffPin] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Tab state
  const [activeTab, setActiveTab] = useState<"protocols" | "models" | "status">("status");

  // Protocol state
  const [protocolStats, setProtocolStats] = useState<ProtocolStats | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<ProtocolSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [newProtocol, setNewProtocol] = useState({
    title: "",
    content: "",
    source: "",
    language: "fr",
    category: "",
  });
  const [isIngesting, setIsIngesting] = useState(false);

  // Model state
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [activeModel, setActiveModel] = useState<string>("");
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [pullingModel, setPullingModel] = useState<string | null>(null);
  const [pullProgress, setPullProgress] = useState<PullProgress | null>(null);

  // System status state
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(false);

  // Error state
  const [error, setError] = useState<string | null>(null);

  // Load data on auth
  useEffect(() => {
    if (isAuthenticated) {
      loadSystemStatus();
      loadProtocolStats();
      loadModels();
    }
  }, [isAuthenticated]);

  // Load system status
  const loadSystemStatus = async () => {
    setIsLoadingStatus(true);
    try {
      const status = await caeApi.admin.getSystemStatus();
      setSystemStatus(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorLoadStatus", lang));
    } finally {
      setIsLoadingStatus(false);
    }
  };

  // Load protocol stats
  const loadProtocolStats = async () => {
    try {
      const stats = await caeApi.admin.getProtocolStats();
      setProtocolStats(stats);
    } catch (err) {
      console.error("Failed to load protocol stats:", err);
    }
  };

  // Search protocols
  const handleSearchProtocols = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    setError(null);
    try {
      const response = await caeApi.admin.searchProtocols(searchQuery);
      setSearchResults(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorLoadProtocols", lang));
    } finally {
      setIsSearching(false);
    }
  };

  // Ingest protocol
  const handleIngestProtocol = async () => {
    if (!newProtocol.title || !newProtocol.content) return;
    setIsIngesting(true);
    setError(null);
    try {
      await caeApi.admin.ingestProtocols([newProtocol], staffPin);
      setNewProtocol({ title: "", content: "", source: "", language: "fr", category: "" });
      loadProtocolStats();
      alert(t("ingestSuccess", lang));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to ingest protocol");
    } finally {
      setIsIngesting(false);
    }
  };

  // Clear protocols
  const handleClearProtocols = async () => {
    if (!confirm(t("clearConfirm", lang))) return;
    try {
      await caeApi.admin.clearProtocols(staffPin);
      loadProtocolStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clear protocols");
    }
  };

  // Load models
  const loadModels = async () => {
    setIsLoadingModels(true);
    try {
      const response = await caeApi.admin.listModels();
      setModels(response.models || []);
      setActiveModel(response.active_model || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorLoadModels", lang));
    } finally {
      setIsLoadingModels(false);
    }
  };

  // Select model
  const handleSelectModel = async (modelId: string) => {
    try {
      await caeApi.admin.selectModel(modelId, staffPin);
      setActiveModel(modelId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to select model");
    }
  };

  // Pull model
  const handlePullModel = async (modelId: string) => {
    setPullingModel(modelId);
    setPullProgress(null);
    try {
      await caeApi.admin.pullModel(modelId, staffPin, (progress) => {
        setPullProgress(progress);
      });
      loadModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to pull model");
    } finally {
      setPullingModel(null);
      setPullProgress(null);
    }
  };

  // Status badge component
  const StatusBadge = ({ status }: { status: string }) => {
    const isReady = status === "ready";
    return (
      <span className={`flex items-center gap-1 text-sm ${
        isReady ? "text-green-600 dark:text-green-400" : "text-gray-500"
      }`}>
        {isReady ? (
          <CheckCircle className="h-4 w-4" />
        ) : (
          <AlertCircle className="h-4 w-4" />
        )}
        {status}
      </span>
    );
  };

  // Auth screen
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="font-light tracking-wide text-center">
              {t("adminDashboard", lang)}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>{t("enterPin", lang)}</Label>
              <Input
                type="password"
                value={staffPin}
                onChange={(e) => setStaffPin(e.target.value)}
                placeholder="****"
              />
            </div>
            <Button
              onClick={() => setIsAuthenticated(true)}
              disabled={!staffPin}
              className="w-full bg-indigo-600 hover:bg-indigo-700"
            >
              {t("login", lang)}
            </Button>
            <Link href={`/?lang=${lang}`} className="block">
              <Button variant="ghost" className="w-full">
                {t("backToHome", lang)}
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

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
              {t("adminDashboard", lang)}
            </h1>
          </div>
          <Button variant="ghost" onClick={() => setIsAuthenticated(false)}>
            {t("logout", lang)}
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
            {error}
            <button onClick={() => setError(null)} className="ml-4 text-sm underline">
              Dismiss
            </button>
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          <Button
            variant={activeTab === "status" ? "default" : "outline"}
            onClick={() => setActiveTab("status")}
          >
            <Settings className="h-4 w-4 mr-2" />
            {t("systemStatus", lang)}
          </Button>
          <Button
            variant={activeTab === "protocols" ? "default" : "outline"}
            onClick={() => setActiveTab("protocols")}
          >
            <Database className="h-4 w-4 mr-2" />
            {t("protocolManagement", lang)}
          </Button>
          <Button
            variant={activeTab === "models" ? "default" : "outline"}
            onClick={() => setActiveTab("models")}
          >
            <Download className="h-4 w-4 mr-2" />
            {t("modelManagement", lang)}
          </Button>
        </div>

        {/* System Status Tab */}
        {activeTab === "status" && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="font-light text-sm">{t("serviceStatus", lang)}</CardTitle>
                <Button variant="ghost" size="sm" onClick={loadSystemStatus} disabled={isLoadingStatus}>
                  <RefreshCw className={`h-4 w-4 ${isLoadingStatus ? "animate-spin" : ""}`} />
                </Button>
              </CardHeader>
              <CardContent className="space-y-3">
                {systemStatus ? (
                  <>
                    <div className="flex justify-between items-center">
                      <span>{t("database", lang)}</span>
                      <StatusBadge status={systemStatus.database} />
                    </div>
                    <div className="flex justify-between items-center">
                      <span>{t("ollama", lang)}</span>
                      <StatusBadge status={systemStatus.ollama} />
                    </div>
                    <div className="flex justify-between items-center">
                      <span>{t("parlant", lang)}</span>
                      <StatusBadge status={systemStatus.parlant} />
                    </div>
                    <div className="flex justify-between items-center">
                      <span>{t("whisper", lang)}</span>
                      <StatusBadge status={systemStatus.whisper} />
                    </div>
                    <div className="flex justify-between items-center">
                      <span>{t("vision", lang)}</span>
                      <StatusBadge status={systemStatus.vision} />
                    </div>
                    <div className="flex justify-between items-center">
                      <span>{t("qdrant", lang)}</span>
                      <StatusBadge status={systemStatus.qdrant} />
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-gray-500">{t("loading", lang)}</p>
                )}
              </CardContent>
            </Card>

            {/* Protocol Stats Card */}
            <Card>
              <CardHeader>
                <CardTitle className="font-light text-sm">{t("protocolStats", lang)}</CardTitle>
              </CardHeader>
              <CardContent>
                {protocolStats ? (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>{t("totalProtocols", lang)}:</span>
                      <span className="font-medium">{protocolStats.total}</span>
                    </div>
                    {Object.entries(protocolStats.by_language).map(([lang, count]) => (
                      <div key={lang} className="flex justify-between text-sm text-gray-500">
                        <span>{lang.toUpperCase()}:</span>
                        <span>{count}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">{t("loading", lang)}</p>
                )}
              </CardContent>
            </Card>

            {/* Active Model Card */}
            <Card>
              <CardHeader>
                <CardTitle className="font-light text-sm">{t("activeModel", lang)}</CardTitle>
              </CardHeader>
              <CardContent>
                {activeModel ? (
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <span className="font-mono text-sm">{activeModel}</span>
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No model selected</p>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Protocols Tab */}
        {activeTab === "protocols" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Search Protocols */}
            <Card>
              <CardHeader>
                <CardTitle className="font-light text-sm">{t("searchProtocols", lang)}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder={t("search", lang)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearchProtocols()}
                  />
                  <Button onClick={handleSearchProtocols} disabled={isSearching}>
                    <Search className="h-4 w-4" />
                  </Button>
                </div>
                {searchResults.length > 0 && (
                  <div className="space-y-2 max-h-80 overflow-y-auto">
                    {searchResults.map((result, i) => (
                      <div key={i} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <div className="font-medium text-sm">{result.title}</div>
                        <div className="text-xs text-gray-500 mt-1">{result.source}</div>
                        <div className="text-xs text-indigo-600 mt-1">
                          {t("relevance", lang)}: {Math.round(result.relevance * 100)}%
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Ingest Protocol */}
            <Card>
              <CardHeader>
                <CardTitle className="font-light text-sm">{t("ingestProtocols", lang)}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label>{t("protocolTitle", lang)}</Label>
                  <Input
                    value={newProtocol.title}
                    onChange={(e) => setNewProtocol({ ...newProtocol, title: e.target.value })}
                  />
                </div>
                <div>
                  <Label>{t("protocolContent", lang)}</Label>
                  <Textarea
                    value={newProtocol.content}
                    onChange={(e) => setNewProtocol({ ...newProtocol, content: e.target.value })}
                    rows={4}
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>{t("protocolSource", lang)}</Label>
                    <Input
                      value={newProtocol.source}
                      onChange={(e) => setNewProtocol({ ...newProtocol, source: e.target.value })}
                    />
                  </div>
                  <div>
                    <Label>{t("language", lang)}</Label>
                    <select
                      value={newProtocol.language}
                      onChange={(e) => setNewProtocol({ ...newProtocol, language: e.target.value })}
                      className="w-full h-10 px-3 rounded-md border bg-background"
                    >
                      <option value="fr">Français</option>
                      <option value="en">English</option>
                    </select>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button onClick={handleIngestProtocol} disabled={isIngesting} className="flex-1">
                    <Plus className="h-4 w-4 mr-2" />
                    {isIngesting ? t("processing", lang) : t("ingestProtocols", lang)}
                  </Button>
                  <Button onClick={handleClearProtocols} variant="destructive">
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Models Tab */}
        {activeTab === "models" && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="font-light text-sm">{t("availableModels", lang)}</CardTitle>
              <Button variant="ghost" size="sm" onClick={loadModels} disabled={isLoadingModels}>
                <RefreshCw className={`h-4 w-4 ${isLoadingModels ? "animate-spin" : ""}`} />
              </Button>
            </CardHeader>
            <CardContent>
              {models.length === 0 ? (
                <p className="text-sm text-gray-500">{t("ollamaNotRunning", lang)}</p>
              ) : (
                <div className="space-y-3">
                  {models.map((model) => (
                    <div
                      key={model.id}
                      className={`p-4 rounded-lg border ${
                        model.id === activeModel
                          ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20"
                          : "border-gray-200 dark:border-gray-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">{model.name}</div>
                          <div className="text-xs text-gray-500">{model.description}</div>
                          <div className="text-xs text-gray-400 mt-1">
                            {model.size_gb.toFixed(1)} GB
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {model.status === "ready" ? (
                            model.id === activeModel ? (
                              <span className="text-xs text-green-600 flex items-center gap-1">
                                <CheckCircle className="h-3 w-3" />
                                Active
                              </span>
                            ) : (
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleSelectModel(model.id)}
                              >
                                {t("selectModel", lang)}
                              </Button>
                            )
                          ) : pullingModel === model.id ? (
                            <div className="text-xs">
                              {pullProgress ? (
                                <span>{Math.round(pullProgress.progress * 100)}%</span>
                              ) : (
                                <RefreshCw className="h-4 w-4 animate-spin" />
                              )}
                            </div>
                          ) : (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handlePullModel(model.id)}
                            >
                              <Download className="h-3 w-3 mr-1" />
                              {t("pullModel", lang)}
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
