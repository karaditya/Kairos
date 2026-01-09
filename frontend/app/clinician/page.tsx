"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  FileText,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Send,
  BookOpen,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { caeApi } from "@/lib/api";
import { t, type LanguageCode } from "@/lib/translations";
import type {
  SessionListItem,
  AgentDraftResponse,
  SyncPreviewResponse,
  MissingField,
  FlaggedField,
  RAGCitation,
} from "@/lib/types";

export default function ClinicianDashboard() {
  const searchParams = useSearchParams();
  const lang = (searchParams.get("lang") || "en") as LanguageCode;

  // State
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [draft, setDraft] = useState<AgentDraftResponse | null>(null);
  const [syncPreview, setSyncPreview] = useState<SyncPreviewResponse | null>(null);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [isGeneratingDraft, setIsGeneratingDraft] = useState(false);
  const [isRequestingSync, setIsRequestingSync] = useState(false);
  const [isApprovingSync, setIsApprovingSync] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [staffPin, setStaffPin] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userId, setUserId] = useState("clinician-1");

  // Load sessions
  const loadSessions = async () => {
    setIsLoadingSessions(true);
    setError(null);
    try {
      const response = await caeApi.session.list();
      setSessions(response.sessions || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorLoadSessions", lang));
    } finally {
      setIsLoadingSessions(false);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      loadSessions();
    }
  }, [isAuthenticated]);

  // Generate draft
  const handleGenerateDraft = async () => {
    if (!selectedSession) return;
    setIsGeneratingDraft(true);
    setError(null);
    try {
      const response = await caeApi.agent.generateDraft(selectedSession);
      setDraft(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorGenerateDraft", lang));
    } finally {
      setIsGeneratingDraft(false);
    }
  };

  // Request sync
  const handleRequestSync = async () => {
    if (!selectedSession || !draft) return;
    setIsRequestingSync(true);
    setError(null);
    try {
      const response = await caeApi.agent.requestSync(
        selectedSession,
        draft.compte_rendu_id,
        staffPin
      );
      setSyncPreview(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorRequestSync", lang));
    } finally {
      setIsRequestingSync(false);
    }
  };

  // Approve sync
  const handleApproveSync = async () => {
    if (!syncPreview) return;
    setIsApprovingSync(true);
    setError(null);
    try {
      await caeApi.agent.approveSync(
        syncPreview.verification_id,
        { user_id: userId },
        staffPin
      );
      setSyncPreview(null);
      setDraft(null);
      alert(t("syncApproved", lang));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("errorApproveSync", lang));
    } finally {
      setIsApprovingSync(false);
    }
  };

  // Reject sync
  const handleRejectSync = async () => {
    if (!syncPreview) return;
    try {
      await caeApi.agent.rejectSync(syncPreview.verification_id, userId, staffPin);
      setSyncPreview(null);
      alert(t("syncRejected", lang));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reject sync");
    }
  };

  // Authentication
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="font-light tracking-wide text-center">
              {t("clinicianDashboard", lang)}
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
              {t("clinicianDashboard", lang)}
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
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sessions List */}
          <Card className="lg:col-span-1">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="font-light text-sm">{t("sessions", lang)}</CardTitle>
              <Button variant="ghost" size="sm" onClick={loadSessions} disabled={isLoadingSessions}>
                <RefreshCw className={`h-4 w-4 ${isLoadingSessions ? "animate-spin" : ""}`} />
              </Button>
            </CardHeader>
            <CardContent>
              {sessions.length === 0 ? (
                <p className="text-sm text-gray-500">{t("noSessions", lang)}</p>
              ) : (
                <div className="space-y-2 max-h-[500px] overflow-y-auto">
                  {sessions.map((session) => (
                    <button
                      key={session.session_id}
                      onClick={() => {
                        setSelectedSession(session.session_id);
                        setDraft(null);
                        setSyncPreview(null);
                      }}
                      className={`w-full p-3 text-left rounded-lg transition-colors ${
                        selectedSession === session.session_id
                          ? "bg-indigo-100 dark:bg-indigo-900"
                          : "bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700"
                      }`}
                    >
                      <div className="font-mono text-xs text-gray-500">
                        {session.session_id.slice(0, 8)}...
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          session.status === "completed" ? "bg-green-100 text-green-800" :
                          session.status === "active" ? "bg-blue-100 text-blue-800" :
                          "bg-gray-100 text-gray-800"
                        }`}>
                          {session.status}
                        </span>
                        <span className="text-xs text-gray-400">
                          {session.language.toUpperCase()}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Compte Rendu Editor */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="font-light flex items-center gap-2">
                <FileText className="h-5 w-5" />
                {t("compteRenduDraft", lang)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!selectedSession ? (
                <p className="text-center py-12 text-gray-500">{t("selectSession", lang)}</p>
              ) : !draft ? (
                <div className="text-center py-12">
                  <Button
                    onClick={handleGenerateDraft}
                    disabled={isGeneratingDraft}
                    className="bg-indigo-600 hover:bg-indigo-700"
                  >
                    {isGeneratingDraft ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        {t("generatingDraft", lang)}
                      </>
                    ) : (
                      t("generateDraft", lang)
                    )}
                  </Button>
                </div>
              ) : (
                <div className="space-y-4 max-h-[600px] overflow-y-auto">
                  {/* Missing Fields Alert */}
                  {draft.compte_rendu.missing_fields.length > 0 && (
                    <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                      <div className="flex items-center gap-2 text-yellow-800 dark:text-yellow-200 font-medium mb-2">
                        <AlertTriangle className="h-4 w-4" />
                        {t("missingFields", lang)}
                      </div>
                      <ul className="text-sm text-yellow-700 dark:text-yellow-300 space-y-1">
                        {draft.compte_rendu.missing_fields.map((field, i) => (
                          <li key={i}>
                            <strong>{field.field_name}</strong>: {field.reason}
                            <span className="text-xs opacity-70"> ({t("requiredFor", lang)}: {field.required_for})</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Flagged Fields */}
                  {draft.compte_rendu.flagged_fields.length > 0 && (
                    <div className="p-3 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg">
                      <div className="flex items-center gap-2 text-orange-800 dark:text-orange-200 font-medium mb-2">
                        <AlertTriangle className="h-4 w-4" />
                        {t("flaggedFields", lang)}
                      </div>
                      <ul className="text-sm text-orange-700 dark:text-orange-300 space-y-1">
                        {draft.compte_rendu.flagged_fields.map((field, i) => (
                          <li key={i}>
                            <strong>{field.field_name}</strong>: {field.reason}
                            {field.requires_clinical_input && (
                              <span className="ml-2 text-xs bg-orange-200 dark:bg-orange-800 px-2 py-0.5 rounded">
                                {t("requiresClinicalInput", lang)}
                              </span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* CR Sections */}
                  <div className="space-y-3">
                    <CompteRenduSection label={t("patientName", lang)} value={draft.compte_rendu.patient_name} />
                    <CompteRenduSection label={t("patientDob", lang)} value={draft.compte_rendu.patient_dob} />
                    <CompteRenduSection label={t("patientMrn", lang)} value={draft.compte_rendu.patient_mrn} />
                    <CompteRenduSection label={t("motifConsultation", lang)} value={draft.compte_rendu.motif_consultation} multiline />
                    <CompteRenduSection label={t("anamnese", lang)} value={draft.compte_rendu.anamnese} multiline />
                    <CompteRenduSection label={t("antecedents", lang)} value={draft.compte_rendu.antecedents} multiline />
                    <CompteRenduSection label={t("allergies", lang)} value={draft.compte_rendu.allergies} />
                    <CompteRenduSection label={t("traitementsActuels", lang)} value={draft.compte_rendu.traitements_actuels} multiline />
                    <CompteRenduSection label={t("examenClinique", lang)} value={draft.compte_rendu.examen_clinique} multiline />
                    <CompteRenduSection label={t("examensComplementaires", lang)} value={draft.compte_rendu.examens_complementaires} multiline />
                    <CompteRenduSection label={t("hypothesesDiagnostiques", lang)} value={draft.compte_rendu.hypotheses_diagnostiques} multiline />
                    <CompteRenduSection label={t("planTherapeutique", lang)} value={draft.compte_rendu.plan_therapeutique} multiline />
                  </div>

                  {/* Disclaimer */}
                  <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg text-xs text-gray-500">
                    {draft.disclaimer}
                  </div>

                  {/* Sync Button */}
                  <Button
                    onClick={handleRequestSync}
                    disabled={isRequestingSync || !!syncPreview}
                    className="w-full bg-indigo-600 hover:bg-indigo-700"
                  >
                    {isRequestingSync ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        {t("processing", lang)}
                      </>
                    ) : (
                      <>
                        <Send className="h-4 w-4 mr-2" />
                        {t("requestSync", lang)}
                      </>
                    )}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Right Panel - RAG Citations & Sync */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="font-light text-sm flex items-center gap-2">
                <BookOpen className="h-4 w-4" />
                {t("ragCitations", lang)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!draft ? (
                <p className="text-sm text-gray-500">{t("noProtocols", lang)}</p>
              ) : draft.rag_citations.length === 0 ? (
                <p className="text-sm text-gray-500">{t("noProtocols", lang)}</p>
              ) : (
                <div className="space-y-3">
                  {draft.rag_citations.map((citation, i) => (
                    <div key={i} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="font-medium text-sm">{citation.protocol}</div>
                      <div className="text-xs text-gray-500 mt-1">{citation.source}</div>
                      <div className="text-xs text-indigo-600 dark:text-indigo-400 mt-1">
                        {t("relevance", lang)}: {Math.round(citation.relevance * 100)}%
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Sync Preview Modal */}
              {syncPreview && (
                <div className="mt-6 p-4 border-2 border-indigo-500 rounded-lg">
                  <h3 className="font-medium text-sm mb-3 flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                    {t("humanVerificationRequired", lang)}
                  </h3>
                  <div className="text-xs text-gray-500 mb-3">
                    {t("verificationId", lang)}: {syncPreview.verification_id.slice(0, 8)}...
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-800 p-2 rounded text-xs font-mono overflow-auto max-h-40 mb-4">
                    {JSON.stringify(syncPreview.preview, null, 2)}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={handleApproveSync}
                      disabled={isApprovingSync}
                      className="flex-1 bg-green-600 hover:bg-green-700"
                      size="sm"
                    >
                      <CheckCircle className="h-4 w-4 mr-1" />
                      {t("approveSync", lang)}
                    </Button>
                    <Button
                      onClick={handleRejectSync}
                      variant="destructive"
                      className="flex-1"
                      size="sm"
                    >
                      <XCircle className="h-4 w-4 mr-1" />
                      {t("rejectSync", lang)}
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

// Helper component for CR sections
function CompteRenduSection({
  label,
  value,
  multiline = false,
}: {
  label: string;
  value?: string;
  multiline?: boolean;
}) {
  if (!value) return null;

  return (
    <div>
      <Label className="text-xs text-gray-500">{label}</Label>
      {multiline ? (
        <div className="mt-1 p-2 bg-gray-50 dark:bg-gray-800 rounded text-sm whitespace-pre-wrap">
          {value}
        </div>
      ) : (
        <div className="mt-1 text-sm font-medium">{value}</div>
      )}
    </div>
  );
}
