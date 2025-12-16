// Translations for English and French medical triage interface

export type LanguageCode = "en" | "fr";

export interface Translations {
  // Common
  backToHome: string;
  continue: string;
  processing: string;
  yes: string;
  no: string;
  print: string;
  returnHome: string;

  // Triage Page
  medicalTriageAssessment: string;
  triageSubtitle: string;
  percentComplete: string;

  // Demographics
  basicInformation: string;
  age: string;
  enterYourAge: string;
  sex: string;
  male: string;
  female: string;
  other: string;
  areYouPregnant: string;
  pleaseFillRequired: string;

  // Chief Complaint
  chiefComplaint: string;
  mainReasonVisit: string;
  selectComplaint: string;
  additionalDetails: string;
  describeSymptoms: string;

  // Summary
  assessmentComplete: string;
  ticketId: string;
  riskLevel: string;
  summary: string;
  keyFlags: string;
  nextSteps: string;
  disclaimer: string;
  generatePdf: string;
  generatingPdf: string;
  printTicket: string;

  // Risk Bands
  red: string;
  amber: string;
  green: string;

  // FRENCH Triage Levels
  triageLevel1: string;
  triageLevel2: string;
  triageLevel3A: string;
  triageLevel3B: string;
  triageLevel4: string;
  triageLevel5: string;

  // Wait Instructions
  waitRed: string;
  waitAmber: string;
  waitGreen: string;

  // FRENCH-specific wait instructions
  waitLevel1: string;
  waitLevel2: string;
  waitLevel3A: string;
  waitLevel3B: string;
  waitLevel4: string;
  waitLevel5: string;

  // Errors
  failedInitSession: string;
  failedSubmitDemographics: string;
  failedSubmitComplaint: string;
  failedSubmitAnswer: string;
  failedGeneratePdf: string;
  provideAnswer: string;

  // Landing Page
  landingTitle: string;
  landingSubtitle: string;
  startTriage: string;
  staffPortal: string;
  trustedBy: string;
  contactUs: string;

  // Staff Portal
  staffLogin: string;
  enterPin: string;
  login: string;
  cases: string;
  pending: string;
  reviewed: string;
  discharged: string;
  searchCases: string;
  caseDetails: string;
  askQuestion: string;
  updateStatus: string;
}

export const translations: Record<LanguageCode, Translations> = {
  en: {
    // Common
    backToHome: "Back to Home",
    continue: "Continue",
    processing: "Processing...",
    yes: "Yes",
    no: "No",
    print: "Print",
    returnHome: "Return Home",

    // Triage Page
    medicalTriageAssessment: "Medical Triage Assessment",
    triageSubtitle: "Please answer the following questions to help us assess your condition",
    percentComplete: "Complete",

    // Demographics
    basicInformation: "Basic Information",
    age: "Age",
    enterYourAge: "Enter your age",
    sex: "Sex",
    male: "Male",
    female: "Female",
    other: "Other",
    areYouPregnant: "Are you pregnant?",
    pleaseFillRequired: "Please fill in all required fields",

    // Chief Complaint
    chiefComplaint: "Chief Complaint",
    mainReasonVisit: "What is the main reason for your visit?",
    selectComplaint: "Please select a chief complaint",
    additionalDetails: "Additional details (optional)",
    describeSymptoms: "Describe your symptoms in more detail...",

    // Summary
    assessmentComplete: "Assessment Complete",
    ticketId: "Ticket ID",
    riskLevel: "Risk Level",
    summary: "Summary",
    keyFlags: "Key Flags",
    nextSteps: "Next Steps",
    disclaimer: "This tool does not provide medical diagnosis or treatment. A healthcare professional will review your case.",
    generatePdf: "Generate Summary PDF",
    generatingPdf: "Generating PDF...",
    printTicket: "Print Ticket",

    // Risk Bands
    red: "RED",
    amber: "AMBER",
    green: "GREEN",

    // FRENCH Triage Levels
    triageLevel1: "Level 1 - Immediate",
    triageLevel2: "Level 2 - Very Urgent",
    triageLevel3A: "Level 3A - Urgent (Priority)",
    triageLevel3B: "Level 3B - Urgent",
    triageLevel4: "Level 4 - Standard",
    triageLevel5: "Level 5 - Non-Urgent",

    // Wait Instructions
    waitRed: "Please proceed immediately to the emergency area. A staff member will assist you.",
    waitAmber: "Please wait in the priority waiting area. You will be seen soon.",
    waitGreen: "Please take a seat in the general waiting area. You will be called when it's your turn.",

    // FRENCH-specific wait instructions
    waitLevel1: "Proceed immediately to the resuscitation room (SAUV). Medical care within 1 minute.",
    waitLevel2: "Proceed to the emergency bay. Medical care within 20 minutes.",
    waitLevel3A: "Priority waiting. Medical care within 60 minutes.",
    waitLevel3B: "Please wait. Medical care within 90 minutes.",
    waitLevel4: "Standard waiting area. Medical care within 2 hours.",
    waitLevel5: "General waiting area or fast track. Medical care within 4 hours.",

    // Errors
    failedInitSession: "Failed to initialize session. Is the backend running?",
    failedSubmitDemographics: "Failed to submit demographics",
    failedSubmitComplaint: "Failed to submit complaint",
    failedSubmitAnswer: "Failed to submit answer",
    failedGeneratePdf: "Failed to generate PDF. Please try again.",
    provideAnswer: "Please provide an answer",

    // Landing Page
    landingTitle: "Kairos",
    landingSubtitle: "AI-Powered Medical Triage",
    startTriage: "Start Triage",
    staffPortal: "Staff Portal",
    trustedBy: "Trusted By",
    contactUs: "Contact Us",

    // Staff Portal
    staffLogin: "Staff Login",
    enterPin: "Enter PIN",
    login: "Login",
    cases: "Cases",
    pending: "Pending",
    reviewed: "Reviewed",
    discharged: "Discharged",
    searchCases: "Search cases...",
    caseDetails: "Case Details",
    askQuestion: "Ask a question about this case...",
    updateStatus: "Update Status",
  },

  fr: {
    // Common
    backToHome: "Retour à l'accueil",
    continue: "Continuer",
    processing: "Traitement en cours...",
    yes: "Oui",
    no: "Non",
    print: "Imprimer",
    returnHome: "Retour à l'accueil",

    // Triage Page
    medicalTriageAssessment: "Évaluation de Triage Médical",
    triageSubtitle: "Veuillez répondre aux questions suivantes pour nous aider à évaluer votre état",
    percentComplete: "Complété",

    // Demographics
    basicInformation: "Informations de Base",
    age: "Âge",
    enterYourAge: "Entrez votre âge",
    sex: "Sexe",
    male: "Homme",
    female: "Femme",
    other: "Autre",
    areYouPregnant: "Êtes-vous enceinte ?",
    pleaseFillRequired: "Veuillez remplir tous les champs obligatoires",

    // Chief Complaint
    chiefComplaint: "Motif de Consultation",
    mainReasonVisit: "Quel est le motif principal de votre consultation ?",
    selectComplaint: "Veuillez sélectionner un motif de consultation",
    additionalDetails: "Détails supplémentaires (optionnel)",
    describeSymptoms: "Décrivez vos symptômes plus en détail...",

    // Summary
    assessmentComplete: "Évaluation Terminée",
    ticketId: "Numéro de ticket",
    riskLevel: "Niveau de Priorité",
    summary: "Résumé",
    keyFlags: "Points Clés",
    nextSteps: "Prochaines Étapes",
    disclaimer: "Cet outil ne fournit pas de diagnostic ou de traitement médical. Un professionnel de santé examinera votre cas.",
    generatePdf: "Générer le PDF",
    generatingPdf: "Génération du PDF...",
    printTicket: "Imprimer le ticket",

    // Risk Bands (mapped to FRENCH levels)
    red: "ROUGE",
    amber: "ORANGE",
    green: "VERT",

    // FRENCH Triage Levels
    triageLevel1: "Tri 1 - Détresse vitale majeure",
    triageLevel2: "Tri 2 - Atteinte patente d'un organe",
    triageLevel3A: "Tri 3A - Urgence avec comorbidité",
    triageLevel3B: "Tri 3B - Urgence sans comorbidité",
    triageLevel4: "Tri 4 - Consultation standard",
    triageLevel5: "Tri 5 - Consultation non urgente",

    // Wait Instructions
    waitRed: "Veuillez vous rendre immédiatement à la salle de déchocage. Un membre du personnel vous assistera.",
    waitAmber: "Veuillez patienter dans la zone d'attente prioritaire. Vous serez pris en charge rapidement.",
    waitGreen: "Veuillez prendre place dans la salle d'attente générale. Vous serez appelé(e) à votre tour.",

    // FRENCH-specific wait instructions
    waitLevel1: "Dirigez-vous immédiatement vers la SAUV (salle d'accueil des urgences vitales). Prise en charge médicale en moins d'1 minute.",
    waitLevel2: "Rendez-vous au box d'examen ou en SAUV. Prise en charge médicale en moins de 20 minutes.",
    waitLevel3A: "Attente prioritaire. Prise en charge médicale en moins de 60 minutes.",
    waitLevel3B: "Veuillez patienter. Prise en charge médicale en moins de 90 minutes.",
    waitLevel4: "Salle d'attente standard. Prise en charge médicale en moins de 2 heures.",
    waitLevel5: "Salle d'attente générale ou circuit court. Prise en charge médicale en moins de 4 heures.",

    // Errors
    failedInitSession: "Échec de l'initialisation de la session. Le serveur est-il en marche ?",
    failedSubmitDemographics: "Échec de l'envoi des informations démographiques",
    failedSubmitComplaint: "Échec de l'envoi du motif de consultation",
    failedSubmitAnswer: "Échec de l'envoi de la réponse",
    failedGeneratePdf: "Échec de la génération du PDF. Veuillez réessayer.",
    provideAnswer: "Veuillez fournir une réponse",

    // Landing Page
    landingTitle: "Kairos",
    landingSubtitle: "Triage Médical assisté par IA",
    startTriage: "Commencer le Triage",
    staffPortal: "Portail Personnel",
    trustedBy: "Ils nous font confiance",
    contactUs: "Contactez-nous",

    // Staff Portal
    staffLogin: "Connexion Personnel",
    enterPin: "Entrez le code PIN",
    login: "Connexion",
    cases: "Cas",
    pending: "En attente",
    reviewed: "Examiné",
    discharged: "Sorti",
    searchCases: "Rechercher des cas...",
    caseDetails: "Détails du cas",
    askQuestion: "Poser une question sur ce cas...",
    updateStatus: "Mettre à jour le statut",
  },
};

// Helper function to get translation
export function t(key: keyof Translations, language: LanguageCode = "en"): string {
  return translations[language][key] || translations.en[key] || key;
}
