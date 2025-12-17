"use client";

import React, { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Lock,
  Search,
  AlertCircle,
  CheckCircle,
  Clock,
  ArrowLeft,
  Cpu,
  Zap,
  HardDrive,
  ChevronDown,
  ChevronRight,
  Loader2,
  Brain,
  MessageSquare,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { GenerateSummaryButton } from "@/components/ui/generate-summary-button";
import { AnimatedDownloadButton } from "@/components/ui/animated-download-button";
import { LanguageSelectorDropdown, type Language } from "@/components/ui/language-selector-dropdown";
import { api, type CaseResponse, type ModelInfo, type CurrentModel } from "@/lib/api";
import Link from "next/link";
import { useRouter } from "next/navigation";

type LanguageCode = "en" | "fr";

const staffTranslations: Record<LanguageCode, Record<string, string>> = {
  en: {
    // Login
    staffPortal: "Staff Portal",
    enterPinPrompt: "Enter your PIN to access the staff portal",
    pin: "PIN",
    enterPinPlaceholder: "Enter PIN (default: 1234)",
    signIn: "Sign In",
    authenticating: "Authenticating...",
    backToHome: "Back to Home",
    invalidPin: "Invalid PIN",
    pleaseEnterPin: "Please enter PIN",
    // Main
    reviewCases: "Review and manage patient triage cases",
    selectAiModel: "Select AI Model",
    chooseModel: "Choose a model for AI-powered responses",
    noModelsAvailable: "No models available. Download models first.",
    modelsNotListed: "Models not listed? Run the download script to add more.",
    active: "Active",
    speed: "Speed",
    refresh: "Refresh",
    signOut: "Sign Out",
    // Cases
    cases: "Cases",
    search: "Search...",
    all: "All",
    red: "Red",
    amber: "Amber",
    green: "Green",
    pending: "Pending",
    reviewed: "Reviewed",
    noCasesFound: "No cases found",
    // Case Details
    demographics: "Demographics",
    age: "Age",
    sex: "Sex",
    pregnant: "Pregnant",
    yes: "Yes",
    no: "No",
    clinicalSummary: "Clinical Summary",
    keyFlags: "Key Flags",
    triggeredRules: "Triggered Rules",
    updateStatus: "Update Status",
    discharge: "Discharge",
    // AI Assistant
    aiAssistant: "AI Assistant",
    askAboutCase: "Ask about this case...",
    ask: "Ask",
    answer: "Answer",
    followUpQuestions: "Follow-up Questions",
    aiReasoning: "AI Reasoning",
    selectCase: "Select a case from the list to view details",
    casesAvailable: "case(s) available",
    failedFetchCases: "Failed to fetch cases",
    failedGetAnswer: "Failed to get answer",
    failedUpdateStatus: "Failed to update case status",
    failedSwitchModel: "Failed to switch model",
    failedGeneratePdf: "Failed to generate PDF. Please try again.",
  },
  fr: {
    // Login
    staffPortal: "Portail Personnel",
    enterPinPrompt: "Entrez votre code PIN pour accéder au portail",
    pin: "PIN",
    enterPinPlaceholder: "Entrez le PIN (par défaut: 1234)",
    signIn: "Connexion",
    authenticating: "Authentification...",
    backToHome: "Retour à l'accueil",
    invalidPin: "PIN invalide",
    pleaseEnterPin: "Veuillez entrer le PIN",
    // Main
    reviewCases: "Examiner et gérer les cas de triage des patients",
    selectAiModel: "Sélectionner le modèle IA",
    chooseModel: "Choisissez un modèle pour les réponses assistées par IA",
    noModelsAvailable: "Aucun modèle disponible. Téléchargez d'abord des modèles.",
    modelsNotListed: "Modèles non listés ? Exécutez le script de téléchargement.",
    active: "Actif",
    speed: "Vitesse",
    refresh: "Actualiser",
    signOut: "Déconnexion",
    // Cases
    cases: "Cas",
    search: "Rechercher...",
    all: "Tous",
    red: "Rouge",
    amber: "Orange",
    green: "Vert",
    pending: "En attente",
    reviewed: "Examiné",
    noCasesFound: "Aucun cas trouvé",
    // Case Details
    demographics: "Données démographiques",
    age: "Âge",
    sex: "Sexe",
    pregnant: "Enceinte",
    yes: "Oui",
    no: "Non",
    clinicalSummary: "Résumé clinique",
    keyFlags: "Points clés",
    triggeredRules: "Règles déclenchées",
    updateStatus: "Mettre à jour le statut",
    discharge: "Sortie",
    // AI Assistant
    aiAssistant: "Assistant IA",
    askAboutCase: "Poser une question sur ce cas...",
    ask: "Demander",
    answer: "Réponse",
    followUpQuestions: "Questions de suivi",
    aiReasoning: "Raisonnement IA",
    selectCase: "Sélectionnez un cas dans la liste pour voir les détails",
    casesAvailable: "cas disponible(s)",
    failedFetchCases: "Échec de récupération des cas",
    failedGetAnswer: "Échec de récupération de la réponse",
    failedUpdateStatus: "Échec de mise à jour du statut",
    failedSwitchModel: "Échec du changement de modèle",
    failedGeneratePdf: "Échec de génération du PDF. Veuillez réessayer.",
  }
};

export default function StaffPortal() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const langParam = searchParams.get("lang");
  const language: LanguageCode = (langParam === "fr" || langParam === "en") ? langParam : "en";

  const t = (key: string): string => {
    return staffTranslations[language]?.[key] || staffTranslations.en[key] || key;
  };

  const handleLanguageChange = (lang: Language) => {
    router.push(`/staff?lang=${lang.code}`);
  };
  const [authenticated, setAuthenticated] = useState(false);
  const [pin, setPin] = useState("");
  const [staffPin, setStaffPin] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const [cases, setCases] = useState<CaseResponse[]>([]);
  const [selectedCase, setSelectedCase] = useState<CaseResponse | null>(null);
  const [filter, setFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  const [askQuestion, setAskQuestion] = useState("");
  const [answer, setAnswer] = useState<{
    answer: string;
    reasoning: string | null;
    has_reasoning: boolean;
    suggested_questions: string[];
    cited_data: string[];
    model_used: string;
    disclaimer: string;
  } | null>(null);
  const [showReasoning, setShowReasoning] = useState(false);

  // Model management state
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [currentModel, setCurrentModel] = useState<CurrentModel | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [modelSwitching, setModelSwitching] = useState(false);

  // PDF generation state
  const [pdfGenerating, setPdfGenerating] = useState(false);
  const [pdfBlob, setPdfBlob] = useState<Blob | null>(null);
  const [showDownloadButton, setShowDownloadButton] = useState(false);

  const handleAuth = async () => {
    if (!pin) {
      setError(t("pleaseEnterPin"));
      return;
    }

    setLoading(true);
    setError("");

    try {
      await api.staffAuth(pin);
      setStaffPin(pin);
      setAuthenticated(true);
    } catch (err) {
      setError(t("invalidPin"));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (authenticated) {
      fetchCases();
      fetchModels();
    }
  }, [authenticated]);

  const fetchModels = async () => {
    try {
      const response = await api.getModels();
      setModels(response.models);
      setCurrentModel(response.current_model);
      if (response.current_model) {
        setSelectedModelId(response.current_model.model_id);
      }
    } catch (err) {
      console.error("Failed to fetch models:", err);
    }
  };

  const handleSwitchModel = async (modelId: string) => {
    if (modelId === currentModel?.model_id) {
      setShowModelSelector(false);
      return;
    }

    setModelSwitching(true);
    setError("");

    try {
      const response = await api.switchModel(modelId, staffPin);
      setCurrentModel(response.model);
      setSelectedModelId(modelId);
      setShowModelSelector(false);
      await fetchModels();
    } catch (err: any) {
      setError(err.message || "Failed to switch model");
    } finally {
      setModelSwitching(false);
    }
  };

  const fetchCases = async () => {
    try {
      const response = await api.getCases(staffPin);
      setCases(response.cases);
    } catch (err) {
      setError(t("failedFetchCases"));
    }
  };

  const handleAskQuestion = async () => {
    if (!askQuestion || !selectedCase) return;

    setLoading(true);
    try {
      const response = await api.askQuestionWithModel(
        selectedCase.id,
        askQuestion,
        staffPin,
        selectedModelId || undefined
      );
      setAnswer(response);
      setAskQuestion("");
    } catch (err) {
      setError(t("failedGetAnswer"));
    } finally {
      setLoading(false);
    }
  };

  const getQualityColor = (rating: number) => {
    if (rating >= 8) return "text-green-600";
    if (rating >= 6) return "text-yellow-600";
    return "text-gray-600";
  };

  const getSpeedColor = (rating: number) => {
    if (rating >= 8) return "text-blue-600";
    if (rating >= 6) return "text-yellow-600";
    return "text-orange-600";
  };

  const handleUpdateStatus = async (caseId: string, status: string) => {
    try {
      await api.updateCaseStatus(caseId, status, staffPin);
      fetchCases();
      if (selectedCase && selectedCase.id === caseId) {
        const updated = await api.getCase(caseId, staffPin);
        setSelectedCase(updated);
      }
    } catch (err) {
      setError(t("failedUpdateStatus"));
    }
  };

  // PDF Generation handlers
  const handleGeneratePDF = async () => {
    if (!selectedCase) return;

    setPdfGenerating(true);
    setError("");
    setPdfBlob(null);
    setShowDownloadButton(false);

    try {
      const blob = await api.generateSummaryPDF(selectedCase.session_id);
      setPdfBlob(blob);
      setShowDownloadButton(true);
    } catch (err) {
      setError(t("failedGeneratePdf"));
    } finally {
      setPdfGenerating(false);
    }
  };

  const handleDownloadPDF = () => {
    if (!pdfBlob || !selectedCase) return;
    const filename = `triage_report_${selectedCase.ticket_id}.pdf`;
    api.downloadBlob(pdfBlob, filename);
  };

  // Reset PDF state when case changes
  React.useEffect(() => {
    setPdfBlob(null);
    setShowDownloadButton(false);
  }, [selectedCase?.id]);

  // Priority mapping for sorting (higher = more urgent)
  const getPriority = (c: CaseResponse): number => {
    const level = c.answers?._triage_level || "5";
    const band = c.risk_band;

    if (language === "fr") {
      // FRENCH system: 1 > 2 > 3A > 3B > 4 > 5
      const frenchPriority: Record<string, number> = {
        "1": 100, "2": 80, "3A": 60, "3B": 50, "4": 30, "5": 10
      };
      return frenchPriority[level] || 10;
    } else {
      // Manchester system: 1 > 2 > 3 > 4 > 5
      const manchesterPriority: Record<string, number> = {
        "1": 100, "2": 80, "3": 60, "4": 40, "5": 20
      };
      return manchesterPriority[level] || 20;
    }
  };

  const filteredCases = cases
    .filter((c) => {
      const level = c.answers?._triage_level || "5";
      const matchesFilter =
        filter === "all" ||
        // Band-based filters
        (filter === "red" && c.risk_band === "red") ||
        (filter === "orange" && c.risk_band === "orange") ||
        (filter === "amber" && c.risk_band === "amber") ||
        (filter === "yellow" && c.risk_band === "yellow") ||
        (filter === "green" && c.risk_band === "green") ||
        (filter === "blue" && c.risk_band === "blue") ||
        // Level-based filters (Manchester)
        (filter === "immediate" && level === "1") ||
        (filter === "veryUrgent" && level === "2") ||
        (filter === "urgent" && level === "3") ||
        (filter === "standard" && level === "4") ||
        (filter === "nonUrgent" && level === "5") ||
        // Level-based filters (FRENCH)
        (filter === "tri1" && level === "1") ||
        (filter === "tri2" && level === "2") ||
        (filter === "tri3A" && level === "3A") ||
        (filter === "tri3B" && level === "3B") ||
        (filter === "tri4" && level === "4") ||
        (filter === "tri5" && level === "5") ||
        // Status filters
        (filter === "pending" && c.status === "pending") ||
        (filter === "reviewed" && c.status === "reviewed");

      const matchesSearch =
        searchQuery === "" ||
        c.ticket_id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        c.summary?.toLowerCase().includes(searchQuery.toLowerCase());

      return matchesFilter && matchesSearch;
    })
    .sort((a, b) => getPriority(b) - getPriority(a)); // Sort by priority (most urgent first)

  // Get color based on triage level and system
  const getRiskColor = (c: CaseResponse): string => {
    const level = c.answers?._triage_level;
    const levelInfo = c.answers?._level_info;

    // If we have level info with color, use it
    if (levelInfo?.color) {
      return levelInfo.color;
    }

    // Otherwise use level-based colors
    if (level) {
      if (language === "fr") {
        // FRENCH system colors
        const frenchColors: Record<string, string> = {
          "1": "#dc3545",  // Red - Life-threatening
          "2": "#ff6b35",  // Orange - Severe
          "3A": "#ffc107", // Yellow - Urgent with comorbidity
          "3B": "#ffda6b", // Light yellow - Urgent
          "4": "#28a745",  // Green - Standard
          "5": "#20c997"   // Teal - Non-urgent
        };
        return frenchColors[level] || "#6c757d";
      } else {
        // Manchester system colors
        const manchesterColors: Record<string, string> = {
          "1": "#dc3545",  // Red - Immediate
          "2": "#ff6b35",  // Orange - Very Urgent
          "3": "#ffc107",  // Yellow - Urgent
          "4": "#28a745",  // Green - Standard
          "5": "#17a2b8"   // Blue - Non-urgent
        };
        return manchesterColors[level] || "#6c757d";
      }
    }

    // Fallback to band-based colors
    const bandColors: Record<string, string> = {
      "red": "#dc3545",
      "orange": "#ff6b35",
      "amber": "#ffc107",
      "yellow": "#ffc107",
      "green": "#28a745",
      "blue": "#17a2b8"
    };
    return bandColors[c.risk_band] || "#6c757d";
  };

  // Get display label for triage level
  const getLevelLabel = (c: CaseResponse): string => {
    const level = c.answers?._triage_level;
    if (!level) return c.risk_band?.toUpperCase() || "N/A";

    if (language === "fr") {
      const frenchLabels: Record<string, string> = {
        "1": "Tri 1", "2": "Tri 2", "3A": "Tri 3A", "3B": "Tri 3B", "4": "Tri 4", "5": "Tri 5"
      };
      return frenchLabels[level] || level;
    } else {
      const manchesterLabels: Record<string, string> = {
        "1": "P1", "2": "P2", "3": "P3", "4": "P4", "5": "P5"
      };
      return manchesterLabels[level] || level;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pending":
        return <Clock className="h-5 w-5 text-yellow-600" />;
      case "reviewed":
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      default:
        return <AlertCircle className="h-5 w-5 text-gray-600" />;
    }
  };

  if (!authenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 flex items-center justify-center p-4">
        {/* Language selector in top-right corner */}
        <div className="absolute top-6 right-30 z-30">
          <LanguageSelectorDropdown
            value={language}
            onChange={handleLanguageChange}
          />
        </div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md"
        >
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2 mb-4">
                <Lock className="h-8 w-8 text-blue-600" />
                <h1 className="text-2xl font-bold">{t("staffPortal")}</h1>
              </div>
              <p className="text-gray-600 dark:text-gray-400">
                {t("enterPinPrompt")}
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              {error && (
                <div className="flex items-center gap-2 text-red-600 bg-red-50 dark:bg-red-900/20 p-3 rounded">
                  <AlertCircle className="h-5 w-5" />
                  <p>{error}</p>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium mb-2">{t("pin")}</label>
                <Input
                  type="password"
                  value={pin}
                  onChange={(e) => setPin(e.target.value)}
                  placeholder={t("enterPinPlaceholder")}
                  onKeyPress={(e) => e.key === "Enter" && handleAuth()}
                />
              </div>

              <Button onClick={handleAuth} disabled={loading} className="w-full">
                {loading ? t("authenticating") : t("signIn")}
              </Button>

              <Link href="/">
                <Button variant="ghost" className="w-full">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  {t("backToHome")}
                </Button>
              </Link>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 py-8 px-6">
      <div className="max-w-[1800px] mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              {t("staffPortal")}
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              {t("reviewCases")}
            </p>
          </div>
          <div className="flex gap-2 items-center mr-14">
            {/* Model Selector */}
            <div className="relative mr-4">
              <Button
                onClick={() => setShowModelSelector(!showModelSelector)}
                variant="outline"
                className="flex items-center gap-2"
                disabled={modelSwitching}
              >
                {modelSwitching ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Cpu className="h-4 w-4" />
                )}
                <span className="max-w-[150px] truncate">
                  {currentModel?.name || "No Model"}
                </span>
                <ChevronDown className="h-4 w-4" />
              </Button>

              {showModelSelector && (
                <div className="absolute right-0 top-full mt-2 w-96 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 z-50 max-h-[500px] overflow-y-auto">
                  <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                    <h3 className="font-semibold text-gray-900 dark:text-white">
                      {t("selectAiModel")}
                    </h3>
                    <p className="text-xs text-gray-500 mt-1">
                      {t("chooseModel")}
                    </p>
                  </div>
                  <div className="p-2">
                    {models.filter(m => m.is_available).length === 0 ? (
                      <p className="text-sm text-gray-500 p-3 text-center">
                        {t("noModelsAvailable")}
                      </p>
                    ) : (
                      models.filter(m => m.is_available).map((model) => (
                        <button
                          key={model.id}
                          onClick={() => handleSwitchModel(model.id)}
                          disabled={modelSwitching}
                          className={`w-full text-left p-3 rounded-lg mb-1 transition-colors ${
                            model.is_loaded
                              ? "bg-blue-50 dark:bg-blue-900/30 border-2 border-blue-500"
                              : "hover:bg-gray-100 dark:hover:bg-gray-700 border-2 border-transparent"
                          }`}
                        >
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-gray-900 dark:text-white">
                                  {model.name}
                                </span>
                                {model.is_loaded && (
                                  <span className="text-xs bg-blue-500 text-white px-2 py-0.5 rounded">
                                    {t("active")}
                                  </span>
                                )}
                              </div>
                              <p className="text-xs text-gray-500 mt-1">
                                {model.description}
                              </p>
                              <div className="flex items-center gap-3 mt-2 text-xs">
                                <span className="flex items-center gap-1">
                                  <HardDrive className="h-3 w-3" />
                                  {model.size_mb}MB
                                </span>
                                <span className={`flex items-center gap-1 ${getSpeedColor(model.speed_rating)}`}>
                                  <Zap className="h-3 w-3" />
                                  {t("speed")}: {model.speed_rating}/10
                                </span>
                                <span className={`flex items-center gap-1 ${getQualityColor(model.quality_rating)}`}>
                                  Quality: {model.quality_rating}/10
                                </span>
                              </div>
                              <div className="flex gap-1 mt-2">
                                {model.tags.slice(0, 3).map((tag) => (
                                  <span
                                    key={tag}
                                    className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded"
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                  <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded-b-lg">
                    <p className="text-xs text-gray-500">
                      {t("modelsNotListed")}
                    </p>
                  </div>
                </div>
              )}
            </div>

            <LanguageSelectorDropdown
              value={language}
              onChange={handleLanguageChange}
            />
            <Button onClick={fetchCases} variant="outline">
              {t("refresh")}
            </Button>
            <Button
              onClick={() => {
                setAuthenticated(false);
                setPin("");
                setStaffPin("");
              }}
              variant="outline"
            >
              {t("signOut")}
            </Button>
          </div>
        </div>

        {error && (
          <Card className="mb-6 border-red-500">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-red-600">
                <AlertCircle className="h-5 w-5" />
                <p>{error}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 3-Column Layout: Cases | Details | AI Assistant */}
        <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr_1fr] gap-6">
          {/* Left Column - Case List (Narrower) */}
          <div className="space-y-4">
            <Card className="h-fit lg:h-[calc(100vh-180px)] flex flex-col">
              <CardHeader className="pb-3">
                <h2 className="text-lg font-semibold">{t("cases")}</h2>
              </CardHeader>
              <CardContent className="space-y-3 flex-1 overflow-hidden flex flex-col">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder={t("search")}
                    className="pl-9 h-9 text-sm"
                  />
                </div>

                {/* Priority Level Filters - Based on triage system */}
                <div className="flex gap-1 flex-wrap">
                  <Button
                    size="sm"
                    onClick={() => setFilter("all")}
                    variant={filter === "all" ? "default" : "outline"}
                    className="h-7 px-2 text-xs"
                  >
                    {t("all")}
                  </Button>
                  {language === "fr" ? (
                    // FRENCH system levels
                    <>
                      {[
                        { key: "tri1", label: "Tri 1", color: "#dc3545" },
                        { key: "tri2", label: "Tri 2", color: "#ff6b35" },
                        { key: "tri3A", label: "3A", color: "#ffc107" },
                        { key: "tri3B", label: "3B", color: "#ffda6b" },
                        { key: "tri4", label: "Tri 4", color: "#28a745" },
                        { key: "tri5", label: "Tri 5", color: "#20c997" },
                      ].map((f) => (
                        <Button
                          key={f.key}
                          size="sm"
                          onClick={() => setFilter(f.key)}
                          variant={filter === f.key ? "default" : "outline"}
                          className="h-7 px-2 text-xs"
                          style={filter === f.key ? { backgroundColor: f.color, borderColor: f.color } : {}}
                        >
                          {f.label}
                        </Button>
                      ))}
                    </>
                  ) : (
                    // Manchester system levels
                    <>
                      {[
                        { key: "immediate", label: "P1", color: "#dc3545" },
                        { key: "veryUrgent", label: "P2", color: "#ff6b35" },
                        { key: "urgent", label: "P3", color: "#ffc107" },
                        { key: "standard", label: "P4", color: "#28a745" },
                        { key: "nonUrgent", label: "P5", color: "#17a2b8" },
                      ].map((f) => (
                        <Button
                          key={f.key}
                          size="sm"
                          onClick={() => setFilter(f.key)}
                          variant={filter === f.key ? "default" : "outline"}
                          className="h-7 px-2 text-xs"
                          style={filter === f.key ? { backgroundColor: f.color, borderColor: f.color } : {}}
                        >
                          {f.label}
                        </Button>
                      ))}
                    </>
                  )}
                </div>
                {/* Status Filters */}
                <div className="flex gap-1 flex-wrap">
                  {["pending", "reviewed"].map((f) => (
                    <Button
                      key={f}
                      size="sm"
                      onClick={() => setFilter(f)}
                      variant={filter === f ? "default" : "outline"}
                      className="h-7 px-2 text-xs"
                    >
                      {t(f)}
                    </Button>
                  ))}
                </div>

                {/* Case List - Scrollable */}
                <div className="space-y-2 flex-1 overflow-y-auto pr-1">
                  {filteredCases.length === 0 ? (
                    <p className="text-gray-500 text-center py-4 text-sm">
                      {t("noCasesFound")}
                    </p>
                  ) : (
                    filteredCases.map((c) => (
                      <Button
                        key={c.id}
                        onClick={() => setSelectedCase(c)}
                        variant={selectedCase?.id === c.id ? "default" : "outline"}
                        className="w-full justify-start h-auto py-2 px-3 flex flex-col items-start text-left"
                      >
                        <div className="flex items-center justify-between w-full mb-0.5">
                          <span className="font-medium text-sm truncate">{c.ticket_id}</span>
                          <div className="flex items-center gap-1.5 flex-shrink-0">
                            {getStatusIcon(c.status)}
                            <span
                              className="text-xs font-semibold px-1.5 py-0.5 rounded text-white"
                              style={{ backgroundColor: getRiskColor(c) }}
                            >
                              {getLevelLabel(c)}
                            </span>
                          </div>
                        </div>
                        <span className="text-xs text-gray-500 truncate w-full">
                          {c.demographics?.age || "?"} y/o {c.demographics?.sex || ""}
                        </span>
                      </Button>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Center + Right Columns */}
          {selectedCase ? (
            <>
              {/* Center Column - Patient Details */}
              <motion.div
                key={`details-${selectedCase.id}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-4"
              >
                <Card className="h-fit lg:h-[calc(100vh-180px)] flex flex-col overflow-hidden">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-lg font-semibold">
                          {selectedCase.ticket_id}
                        </h2>
                        <p className="text-xs text-gray-500">
                          {new Date(selectedCase.created_at).toLocaleString()}
                        </p>
                      </div>
                      <div
                        className="px-3 py-1 rounded-full text-white text-sm font-semibold"
                        style={{ backgroundColor: getRiskColor(selectedCase) }}
                      >
                        {getLevelLabel(selectedCase)}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4 flex-1 overflow-y-auto">
                    {/* Demographics */}
                    <div>
                      <h3 className="font-semibold text-sm mb-2">{t("demographics")}</h3>
                      <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded text-sm">
                        <p>{t("age")}: {selectedCase.demographics?.age || "N/A"}</p>
                        <p className="capitalize">{t("sex")}: {selectedCase.demographics?.sex || "N/A"}</p>
                        {selectedCase.demographics?.pregnant !== undefined && (
                          <p>{t("pregnant")}: {selectedCase.demographics.pregnant ? t("yes") : t("no")}</p>
                        )}
                      </div>
                    </div>

                    {/* Summary */}
                    <div>
                      <h3 className="font-semibold text-sm mb-2">{t("clinicalSummary")}</h3>
                      <p className="text-sm text-gray-700 dark:text-gray-300">
                        {selectedCase.summary}
                      </p>
                    </div>

                    {/* Key Flags */}
                    {selectedCase.key_flags && selectedCase.key_flags.length > 0 && (
                      <div>
                        <h3 className="font-semibold text-sm mb-2">{t("keyFlags")}</h3>
                        <ul className="list-disc list-inside space-y-0.5 text-sm">
                          {selectedCase.key_flags.map((flag, index) => (
                            <li key={index} className="text-gray-700 dark:text-gray-300">
                              {flag}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Triggered Rules */}
                    {selectedCase.triggered_rules && selectedCase.triggered_rules.length > 0 && (
                      <div>
                        <h3 className="font-semibold text-sm mb-2">{t("triggeredRules")}</h3>
                        <div className="space-y-2">
                          {selectedCase.triggered_rules.map((rule: any, index: number) => (
                            <div
                              key={index}
                              className="bg-red-50 dark:bg-red-900/20 p-2 rounded border border-red-200 dark:border-red-800"
                            >
                              <p className="font-medium text-sm">{rule.id}</p>
                              <p className="text-xs text-gray-600 dark:text-gray-400">
                                {rule.description}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Status Update */}
                    <div className="border-t pt-4">
                      <h3 className="font-semibold text-sm mb-3">{t("updateStatus")}</h3>
                      <div className="flex gap-2">
                        <Button
                          onClick={() => handleUpdateStatus(selectedCase.id, "reviewed")}
                          variant="outline"
                          size="sm"
                          className="flex-1"
                          disabled={selectedCase.status === "reviewed"}
                        >
                          <CheckCircle className="mr-1.5 h-3.5 w-3.5" />
                          {t("reviewed")}
                        </Button>
                        <Button
                          onClick={() => handleUpdateStatus(selectedCase.id, "discharged")}
                          variant="outline"
                          size="sm"
                          className="flex-1"
                        >
                          {t("discharge")}
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Right Column - AI Assistant */}
              <motion.div
                key={`ai-${selectedCase.id}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="space-y-4"
              >
                <Card className="h-fit lg:h-[calc(100vh-180px)] flex flex-col overflow-hidden">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <h2 className="text-lg font-semibold flex items-center gap-2">
                          <Brain className="h-5 w-5 text-purple-600" />
                          {t("aiAssistant")}
                        </h2>
                      </div>
                      {currentModel && (
                        <span className="text-xs text-gray-500 flex items-center gap-1">
                          <Cpu className="h-3 w-3" />
                          {currentModel.name}
                        </span>
                      )}
                    </div>
                    {/* Generate Summary / Download PDF Buttons */}
                    <div className="flex items-center gap-2 mt-4 mb-2">
                      <motion.div
                        animate={{
                          opacity: showDownloadButton ? 0 : 1,
                          width: showDownloadButton ? 0 : "auto",
                        }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                      >
                        <GenerateSummaryButton
                          onClick={handleGeneratePDF}
                          isLoading={pdfGenerating}
                          disabled={pdfGenerating}
                        />
                      </motion.div>
                      {showDownloadButton && (
                        <AnimatedDownloadButton
                          onClick={handleDownloadPDF}
                          filename={`triage_report_${selectedCase.ticket_id}.pdf`}
                        />
                      )}
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4 flex-1 overflow-y-auto pt-4">
                    {/* Ask Question */}
                    <div className="flex gap-2">
                      <Input
                        value={askQuestion}
                        onChange={(e) => setAskQuestion(e.target.value)}
                        placeholder={t("askAboutCase")}
                        className="h-9 text-sm"
                        onKeyDown={(e) => e.key === "Enter" && handleAskQuestion()}
                      />
                      <Button
                        onClick={handleAskQuestion}
                        disabled={loading || !askQuestion}
                        size="sm"
                        className="px-3"
                      >
                        {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : t("ask")}
                      </Button>
                    </div>

                    {/* AI Response */}
                    {answer && (
                      <div className="space-y-3">
                        {/* Main Answer */}
                        {answer.answer && answer.answer.trim() && (
                          <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                            <div className="flex items-start gap-2">
                              <MessageSquare className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                              <div className="flex-1">
                                <h4 className="text-xs font-semibold text-blue-700 dark:text-blue-300 mb-1">
                                  {t("answer")}
                                </h4>
                                <div className="text-sm text-gray-800 dark:text-gray-200 leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1.5">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {answer.answer}
                                  </ReactMarkdown>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Suggested Follow-up Questions */}
                        {answer.suggested_questions && answer.suggested_questions.length > 0 && (
                          <div className="bg-green-50 dark:bg-green-900/20 p-3 rounded-lg border border-green-200 dark:border-green-800">
                            <h4 className="text-xs font-semibold text-green-700 dark:text-green-300 mb-2 flex items-center gap-1">
                              <AlertCircle className="h-3.5 w-3.5" />
                              {t("followUpQuestions")}
                            </h4>
                            <ul className="space-y-1.5">
                              {answer.suggested_questions.map((q, idx) => (
                                <li
                                  key={idx}
                                  className="flex items-start gap-2 text-xs text-gray-700 dark:text-gray-300"
                                >
                                  <span className="flex-shrink-0 w-4 h-4 rounded-full bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200 text-xs flex items-center justify-center font-medium">
                                    {idx + 1}
                                  </span>
                                  <span>{q}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Chain of Thought (Collapsible) */}
                        {answer.reasoning && answer.reasoning.trim().length > 0 && (
                          <div className="border border-purple-200 dark:border-purple-800 rounded-lg overflow-hidden">
                            <button
                              onClick={() => setShowReasoning(!showReasoning)}
                              className="w-full flex items-center gap-2 p-2 bg-purple-50 dark:bg-purple-900/20 hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
                            >
                              {showReasoning ? (
                                <ChevronDown className="h-3.5 w-3.5 text-purple-600" />
                              ) : (
                                <ChevronRight className="h-3.5 w-3.5 text-purple-600" />
                              )}
                              <Brain className="h-3.5 w-3.5 text-purple-600" />
                              <span className="text-xs font-medium text-purple-700 dark:text-purple-300">
                                {t("aiReasoning")}
                              </span>
                            </button>
                            {showReasoning && (
                              <div className="p-3 bg-purple-50/50 dark:bg-purple-900/10 border-t border-purple-200 dark:border-purple-800">
                                <div className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed prose prose-sm dark:prose-invert max-w-none">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {answer.reasoning}
                                  </ReactMarkdown>
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Model & Disclaimer */}
                        <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-200 dark:border-gray-700">
                          <span className="flex items-center gap-1">
                            <Cpu className="h-3 w-3" />
                            {answer.model_used}
                          </span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </>
          ) : (
            /* No Case Selected - Spanning center and right */
            <div className="lg:col-span-2">
              <Card className="h-fit lg:h-[calc(100vh-180px)] flex items-center justify-center">
                <CardContent className="py-12 text-center">
                  <div className="text-gray-400 mb-4">
                    <Search className="h-12 w-12 mx-auto opacity-50" />
                  </div>
                  <p className="text-gray-500 text-lg font-medium">
                    {t("selectCase")}
                  </p>
                  <p className="text-gray-400 text-sm mt-2">
                    {filteredCases.length} {t("casesAvailable")}
                  </p>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
