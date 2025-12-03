"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowRight, CheckCircle, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { api, type SummaryResponse } from "@/lib/api";
import Link from "next/link";

type Step = "demographics" | "complaint" | "triage" | "summary";

export default function TriagePage() {
  const [step, setStep] = useState<Step>("demographics");
  const [sessionId, setSessionId] = useState<string>("");
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  // Demographics
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  const [pregnant, setPregnant] = useState<boolean | undefined>(undefined);

  // Complaint
  const [complaints, setComplaints] = useState<any[]>([]);
  const [selectedComplaint, setSelectedComplaint] = useState("");
  const [complaintText, setComplaintText] = useState("");

  // Triage
  const [currentQuestion, setCurrentQuestion] = useState<any>(null);
  const [currentAnswer, setCurrentAnswer] = useState<any>("");

  // Summary
  const [summary, setSummary] = useState<SummaryResponse | null>(null);

  // Initialize session on mount
  useEffect(() => {
    const initSession = async () => {
      try {
        const response = await api.startSession("en");
        setSessionId(response.session_id);
        setProgress(response.progress);
      } catch (err) {
        setError("Failed to initialize session. Is the backend running?");
      }
    };
    initSession();
  }, []);

  const handleDemographicsSubmit = async () => {
    if (!age || !sex) {
      setError("Please fill in all required fields");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await api.submitDemographics(sessionId, {
        age: parseInt(age),
        sex,
        pregnant: sex === "female" ? pregnant : undefined,
      });

      setComplaints(response.next_question?.options || []);
      setProgress(response.progress);
      setStep("complaint");
    } catch (err) {
      setError("Failed to submit demographics");
    } finally {
      setLoading(false);
    }
  };

  const handleComplaintSubmit = async () => {
    if (!selectedComplaint) {
      setError("Please select a chief complaint");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await api.submitComplaint(sessionId, {
        complaint_id: selectedComplaint,
        free_text: complaintText,
      });

      setCurrentQuestion(response.next_question);
      setProgress(response.progress);
      setStep("triage");
    } catch (err) {
      setError("Failed to submit complaint");
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerSubmit = async () => {
    if (currentAnswer === "" || currentAnswer === null) {
      setError("Please provide an answer");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await api.submitAnswer(
        sessionId,
        currentQuestion.id,
        currentAnswer
      );

      setProgress(response.progress);

      if (response.is_complete) {
        // Fetch summary
        const summaryData = await api.getSummary(sessionId);
        setSummary(summaryData);
        setStep("summary");
      } else {
        setCurrentQuestion(response.next_question);
        setCurrentAnswer("");
      }
    } catch (err) {
      setError("Failed to submit answer");
    } finally {
      setLoading(false);
    }
  };

  const renderQuestion = () => {
    if (!currentQuestion) return null;

    switch (currentQuestion.type) {
      case "yesno":
        return (
          <div className="space-y-4">
            <p className="text-lg mb-6">{currentQuestion.text}</p>
            <div className="flex gap-4">
              <Button
                onClick={() => setCurrentAnswer(true)}
                variant={currentAnswer === true ? "default" : "outline"}
                className="flex-1"
              >
                Yes
              </Button>
              <Button
                onClick={() => setCurrentAnswer(false)}
                variant={currentAnswer === false ? "default" : "outline"}
                className="flex-1"
              >
                No
              </Button>
            </div>
          </div>
        );

      case "choice":
        return (
          <div className="space-y-4">
            <p className="text-lg mb-6">{currentQuestion.text}</p>
            <div className="space-y-2">
              {currentQuestion.options?.map((option: any, index: number) => (
                <Button
                  key={index}
                  onClick={() => setCurrentAnswer(option.id || option)}
                  variant={
                    currentAnswer === (option.id || option)
                      ? "default"
                      : "outline"
                  }
                  className="w-full justify-start"
                >
                  {option.name || option.label || option}
                </Button>
              ))}
            </div>
          </div>
        );

      case "numeric":
        return (
          <div className="space-y-4">
            <p className="text-lg mb-6">{currentQuestion.text}</p>
            <Input
              type="number"
              value={currentAnswer}
              onChange={(e) => setCurrentAnswer(parseFloat(e.target.value))}
              placeholder="Enter a number"
            />
          </div>
        );

      case "text":
        return (
          <div className="space-y-4">
            <p className="text-lg mb-6">{currentQuestion.text}</p>
            <Input
              type="text"
              value={currentAnswer}
              onChange={(e) => setCurrentAnswer(e.target.value)}
              placeholder="Type your answer"
            />
          </div>
        );

      default:
        return <p>Unknown question type</p>;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Link href="/">
            <Button variant="ghost" className="mb-4">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Button>
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Medical Triage Assessment
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Please answer the following questions to help us assess your condition
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-blue-600"
              initial={{ width: 0 }}
              animate={{ width: `${progress * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
            {Math.round(progress * 100)}% Complete
          </p>
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

        {/* Demographics Step */}
        {step === "demographics" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card>
              <CardHeader>
                <h2 className="text-2xl font-semibold">Basic Information</h2>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Age *
                  </label>
                  <Input
                    type="number"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    placeholder="Enter your age"
                    min="0"
                    max="120"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Sex *
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {["male", "female", "other"].map((option) => (
                      <Button
                        key={option}
                        onClick={() => setSex(option)}
                        variant={sex === option ? "default" : "outline"}
                        className="capitalize"
                      >
                        {option}
                      </Button>
                    ))}
                  </div>
                </div>

                {sex === "female" && (
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Are you pregnant?
                    </label>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => setPregnant(true)}
                        variant={pregnant === true ? "default" : "outline"}
                        className="flex-1"
                      >
                        Yes
                      </Button>
                      <Button
                        onClick={() => setPregnant(false)}
                        variant={pregnant === false ? "default" : "outline"}
                        className="flex-1"
                      >
                        No
                      </Button>
                    </div>
                  </div>
                )}

                <Button
                  onClick={handleDemographicsSubmit}
                  disabled={loading}
                  className="w-full"
                >
                  {loading ? "Processing..." : "Continue"}
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Complaint Step */}
        {step === "complaint" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card>
              <CardHeader>
                <h2 className="text-2xl font-semibold">Chief Complaint</h2>
                <p className="text-gray-600 dark:text-gray-400">
                  What is the main reason for your visit?
                </p>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-2">
                  {complaints.map((complaint: any, index: number) => (
                    <Button
                      key={index}
                      onClick={() =>
                        setSelectedComplaint(complaint.id || complaint)
                      }
                      variant={
                        selectedComplaint === (complaint.id || complaint)
                          ? "default"
                          : "outline"
                      }
                      className="justify-start text-left h-auto py-4"
                    >
                      {complaint.name || complaint.label || complaint}
                    </Button>
                  ))}
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Additional details (optional)
                  </label>
                  <textarea
                    value={complaintText}
                    onChange={(e) => setComplaintText(e.target.value)}
                    className="w-full min-h-[100px] px-3 py-2 border border-input rounded-md bg-background"
                    placeholder="Describe your symptoms in more detail..."
                  />
                </div>

                <Button
                  onClick={handleComplaintSubmit}
                  disabled={loading || !selectedComplaint}
                  className="w-full"
                >
                  {loading ? "Processing..." : "Continue"}
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Triage Step */}
        {step === "triage" && currentQuestion && (
          <motion.div
            key={currentQuestion.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card>
              <CardContent className="pt-6 space-y-6">
                {renderQuestion()}

                <Button
                  onClick={handleAnswerSubmit}
                  disabled={loading || currentAnswer === "" || currentAnswer === null}
                  className="w-full"
                >
                  {loading ? "Processing..." : "Continue"}
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Summary Step */}
        {step === "summary" && summary && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card
              className="border-4"
              style={{ borderColor: summary.risk_color }}
            >
              <CardHeader>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-8 w-8 text-green-600" />
                  <div>
                    <h2 className="text-2xl font-semibold">Assessment Complete</h2>
                    <p className="text-gray-600 dark:text-gray-400">
                      Ticket ID: {summary.ticket_id}
                    </p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="font-semibold mb-2">Risk Level</h3>
                  <div
                    className="inline-block px-4 py-2 rounded-full text-white font-semibold uppercase"
                    style={{ backgroundColor: summary.risk_color }}
                  >
                    {summary.risk_band}
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-2">Summary</h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    {summary.summary}
                  </p>
                </div>

                {summary.key_flags && summary.key_flags.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-2">Key Flags</h3>
                    <ul className="list-disc list-inside space-y-1">
                      {summary.key_flags.map((flag, index) => (
                        <li key={index} className="text-gray-700 dark:text-gray-300">
                          {flag}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">Next Steps</h3>
                  <p className="text-gray-700 dark:text-gray-300">
                    {summary.waiting_instruction}
                  </p>
                </div>

                <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-800">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    ⚠️ {summary.disclaimer}
                  </p>
                </div>

                <div className="flex gap-4">
                  <Button onClick={() => window.print()} variant="outline" className="flex-1">
                    Print Ticket
                  </Button>
                  <Link href="/" className="flex-1">
                    <Button className="w-full">Return Home</Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
}
