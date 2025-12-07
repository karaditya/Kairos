"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
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
import { api, type CaseResponse, type ModelInfo, type CurrentModel } from "@/lib/api";
import Link from "next/link";

export default function StaffPortal() {
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

  const handleAuth = async () => {
    if (!pin) {
      setError("Please enter PIN");
      return;
    }

    setLoading(true);
    setError("");

    try {
      await api.staffAuth(pin);
      setStaffPin(pin);
      setAuthenticated(true);
    } catch (err) {
      setError("Invalid PIN");
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
      setError("Failed to fetch cases");
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
      setError("Failed to get answer");
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
      setError("Failed to update case status");
    }
  };

  const filteredCases = cases.filter((c) => {
    const matchesFilter =
      filter === "all" ||
      (filter === "red" && c.risk_band === "red") ||
      (filter === "amber" && c.risk_band === "amber") ||
      (filter === "green" && c.risk_band === "green") ||
      (filter === "pending" && c.status === "pending") ||
      (filter === "reviewed" && c.status === "reviewed");

    const matchesSearch =
      searchQuery === "" ||
      c.ticket_id?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      c.summary?.toLowerCase().includes(searchQuery.toLowerCase());

    return matchesFilter && matchesSearch;
  });

  const getRiskColor = (band: string) => {
    switch (band) {
      case "red":
        return "#dc3545";
      case "amber":
        return "#ffc107";
      case "green":
        return "#28a745";
      default:
        return "#6c757d";
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
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md"
        >
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2 mb-4">
                <Lock className="h-8 w-8 text-blue-600" />
                <h1 className="text-2xl font-bold">Staff Portal</h1>
              </div>
              <p className="text-gray-600 dark:text-gray-400">
                Enter your PIN to access the staff portal
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
                <label className="block text-sm font-medium mb-2">PIN</label>
                <Input
                  type="password"
                  value={pin}
                  onChange={(e) => setPin(e.target.value)}
                  placeholder="Enter PIN (default: 1234)"
                  onKeyPress={(e) => e.key === "Enter" && handleAuth()}
                />
              </div>

              <Button onClick={handleAuth} disabled={loading} className="w-full">
                {loading ? "Authenticating..." : "Sign In"}
              </Button>

              <Link href="/">
                <Button variant="ghost" className="w-full">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Home
                </Button>
              </Link>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
              Staff Portal
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Review and manage patient triage cases
            </p>
          </div>
          <div className="flex gap-2 items-center">
            {/* Model Selector */}
            <div className="relative">
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
                      Select AI Model
                    </h3>
                    <p className="text-xs text-gray-500 mt-1">
                      Choose a model for AI-powered responses
                    </p>
                  </div>
                  <div className="p-2">
                    {models.filter(m => m.is_available).length === 0 ? (
                      <p className="text-sm text-gray-500 p-3 text-center">
                        No models available. Download models first.
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
                                    Active
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
                                  Speed: {model.speed_rating}/10
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
                      Models not listed? Run the download script to add more.
                    </p>
                  </div>
                </div>
              )}
            </div>

            <Button onClick={fetchCases} variant="outline">
              Refresh
            </Button>
            <Button
              onClick={() => {
                setAuthenticated(false);
                setPin("");
                setStaffPin("");
              }}
              variant="outline"
            >
              Sign Out
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

        <div className="grid md:grid-cols-3 gap-6">
          {/* Case List */}
          <div className="md:col-span-1 space-y-4">
            <Card>
              <CardHeader>
                <h2 className="text-xl font-semibold">Cases</h2>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search by ticket ID..."
                    className="pl-10"
                  />
                </div>

                {/* Filters */}
                <div className="flex gap-2 flex-wrap">
                  {["all", "red", "amber", "green", "pending", "reviewed"].map(
                    (f) => (
                      <Button
                        key={f}
                        size="sm"
                        onClick={() => setFilter(f)}
                        variant={filter === f ? "default" : "outline"}
                        className="capitalize"
                      >
                        {f}
                      </Button>
                    )
                  )}
                </div>

                {/* Case List */}
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {filteredCases.length === 0 ? (
                    <p className="text-gray-500 text-center py-4">
                      No cases found
                    </p>
                  ) : (
                    filteredCases.map((c) => (
                      <Button
                        key={c.id}
                        onClick={() => setSelectedCase(c)}
                        variant={
                          selectedCase?.id === c.id ? "default" : "outline"
                        }
                        className="w-full justify-start h-auto py-3 flex flex-col items-start"
                      >
                        <div className="flex items-center justify-between w-full mb-1">
                          <span className="font-semibold">{c.ticket_id}</span>
                          <div className="flex items-center gap-2">
                            {getStatusIcon(c.status)}
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: getRiskColor(c.risk_band) }}
                            />
                          </div>
                        </div>
                        <span className="text-xs text-gray-500">
                          Age: {c.demographics?.age || "N/A"} | Sex: {c.demographics?.sex || "N/A"}
                        </span>
                        <span className="text-xs text-gray-500">
                          {new Date(c.created_at).toLocaleString()}
                        </span>
                      </Button>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Case Details */}
          <div className="md:col-span-2">
            {selectedCase ? (
              <motion.div
                key={selectedCase.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-2xl font-semibold">
                          Case Details: {selectedCase.ticket_id}
                        </h2>
                        <p className="text-gray-600 dark:text-gray-400">
                          {new Date(selectedCase.created_at).toLocaleString()}
                        </p>
                      </div>
                      <div
                        className="px-4 py-2 rounded-full text-white font-semibold uppercase"
                        style={{
                          backgroundColor: getRiskColor(selectedCase.risk_band),
                        }}
                      >
                        {selectedCase.risk_band}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Demographics */}
                    <div>
                      <h3 className="font-semibold mb-2">Demographics</h3>
                      <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded">
                        <p>Age: {selectedCase.demographics?.age || "N/A"}</p>
                        <p className="capitalize">Sex: {selectedCase.demographics?.sex || "N/A"}</p>
                        {selectedCase.demographics?.pregnant !== undefined && (
                          <p>
                            Pregnant: {selectedCase.demographics.pregnant ? "Yes" : "No"}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Summary */}
                    <div>
                      <h3 className="font-semibold mb-2">Clinical Summary</h3>
                      <p className="text-gray-700 dark:text-gray-300">
                        {selectedCase.summary}
                      </p>
                    </div>

                    {/* Key Flags */}
                    {selectedCase.key_flags && selectedCase.key_flags.length > 0 && (
                      <div>
                        <h3 className="font-semibold mb-2">Key Flags</h3>
                        <ul className="list-disc list-inside space-y-1">
                          {selectedCase.key_flags.map((flag, index) => (
                            <li
                              key={index}
                              className="text-gray-700 dark:text-gray-300"
                            >
                              {flag}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Triggered Rules */}
                    {selectedCase.triggered_rules && selectedCase.triggered_rules.length > 0 && (
                      <div>
                        <h3 className="font-semibold mb-2">Triggered Rules</h3>
                        <div className="space-y-2">
                          {selectedCase.triggered_rules.map((rule: any, index: number) => (
                            <div
                              key={index}
                              className="bg-red-50 dark:bg-red-900/20 p-3 rounded border border-red-200 dark:border-red-800"
                            >
                              <p className="font-medium">{rule.id}</p>
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                {rule.description}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* AI Assistant */}
                    <div className="border-t pt-6">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold">AI Assistant</h3>
                        {currentModel && (
                          <span className="text-xs text-gray-500 flex items-center gap-1">
                            <Cpu className="h-3 w-3" />
                            Using: {currentModel.name}
                          </span>
                        )}
                      </div>
                      <div className="space-y-4">
                        <div className="flex gap-2">
                          <Input
                            value={askQuestion}
                            onChange={(e) => setAskQuestion(e.target.value)}
                            placeholder="Ask a question about this case..."
                            onKeyPress={(e) =>
                              e.key === "Enter" && handleAskQuestion()
                            }
                          />
                          <Button
                            onClick={handleAskQuestion}
                            disabled={loading || !askQuestion}
                          >
                            {loading ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              "Ask"
                            )}
                          </Button>
                        </div>

                        {answer && (
                          <div className="space-y-3">
                            {/* Main Answer */}
                            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                              <div className="flex items-start gap-3">
                                <MessageSquare className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                                <div className="flex-1">
                                  <h4 className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-2">
                                    Answer
                                  </h4>
                                  <div className="text-gray-800 dark:text-gray-200 leading-relaxed prose prose-sm dark:prose-invert max-w-none">
                                    <ReactMarkdown>
                                      {answer.answer || "No specific answer generated."}
                                    </ReactMarkdown>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* Suggested Follow-up Questions */}
                            {answer.suggested_questions && answer.suggested_questions.length > 0 && (
                              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                                <h4 className="text-sm font-semibold text-green-700 dark:text-green-300 mb-3 flex items-center gap-2">
                                  <AlertCircle className="h-4 w-4" />
                                  Suggested Follow-up Questions
                                </h4>
                                <ul className="space-y-2">
                                  {answer.suggested_questions.map((q, idx) => (
                                    <li
                                      key={idx}
                                      className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300"
                                    >
                                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200 text-xs flex items-center justify-center font-medium">
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
                                  className="w-full flex items-center gap-2 p-3 bg-purple-50 dark:bg-purple-900/20 hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
                                >
                                  {showReasoning ? (
                                    <ChevronDown className="h-4 w-4 text-purple-600" />
                                  ) : (
                                    <ChevronRight className="h-4 w-4 text-purple-600" />
                                  )}
                                  <Brain className="h-4 w-4 text-purple-600" />
                                  <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
                                    View AI Reasoning Process
                                  </span>
                                  <span className="text-xs text-purple-500 ml-auto">
                                    Chain-of-Thought
                                  </span>
                                </button>
                                {showReasoning && (
                                  <div className="p-4 bg-purple-50/50 dark:bg-purple-900/10 border-t border-purple-200 dark:border-purple-800">
                                    <div className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed prose prose-sm dark:prose-invert max-w-none">
                                      <ReactMarkdown>
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
                              <span className="text-right max-w-[60%]">{answer.disclaimer}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Status Update */}
                    <div className="border-t pt-6">
                      <h3 className="font-semibold mb-4">Update Status</h3>
                      <div className="flex gap-2">
                        <Button
                          onClick={() =>
                            handleUpdateStatus(selectedCase.id, "reviewed")
                          }
                          variant="outline"
                          className="flex-1"
                          disabled={selectedCase.status === "reviewed"}
                        >
                          <CheckCircle className="mr-2 h-4 w-4" />
                          Mark as Reviewed
                        </Button>
                        <Button
                          onClick={() =>
                            handleUpdateStatus(selectedCase.id, "discharged")
                          }
                          variant="outline"
                          className="flex-1"
                        >
                          Discharge
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ) : (
              <Card>
                <CardContent className="py-12 text-center">
                  <p className="text-gray-500">
                    Select a case from the list to view details
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
