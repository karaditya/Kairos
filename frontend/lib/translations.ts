// =============================================================================
// Clinical Admin Edge (CAE) System - Translations
// Bilingual support for English and French
// =============================================================================

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
  save: string;
  cancel: string;
  delete: string;
  edit: string;
  search: string;
  loading: string;
  error: string;
  success: string;
  confirm: string;
  close: string;
  refresh: string;

  // Landing Page
  landingTitle: string;
  landingSubtitle: string;
  receptionistPortal: string;
  clinicianPortal: string;
  adminPortal: string;
  trustedBy: string;
  contactUs: string;
  getInTouch: string;
  contactDescription: string;
  email: string;
  phone: string;
  address: string;
  name: string;
  yourName: string;
  yourEmail: string;
  message: string;
  howCanWeHelp: string;
  submit: string;

  // Authentication
  staffLogin: string;
  enterPin: string;
  login: string;
  logout: string;
  invalidPin: string;

  // Receptionist Dashboard
  receptionistDashboard: string;
  newSession: string;
  startSession: string;
  endSession: string;
  pauseSession: string;
  resumeSession: string;
  sessionActive: string;
  sessionPaused: string;
  sessionEnded: string;
  sessionId: string;
  patientId: string;
  language: string;
  selectLanguage: string;
  french: string;
  english: string;

  // Audio
  audioRecording: string;
  startRecording: string;
  stopRecording: string;
  uploadAudio: string;
  dragDropAudio: string;
  supportedFormats: string;
  transcribing: string;
  transcriptEmpty: string;

  // Transcript
  liveTranscript: string;
  transcript: string;
  confidence: string;
  keywords: string;
  noKeywords: string;
  speaker: string;
  timestamp: string;

  // Clinician Dashboard
  clinicianDashboard: string;
  sessions: string;
  selectSession: string;
  noSessions: string;
  sessionDetails: string;
  generateDraft: string;
  generatingDraft: string;

  // Compte Rendu
  compteRendu: string;
  compteRenduDraft: string;
  patientName: string;
  patientDob: string;
  patientMrn: string;
  motifConsultation: string;
  anamnese: string;
  antecedents: string;
  allergies: string;
  traitementsActuels: string;
  examenClinique: string;
  examensComplementaires: string;
  hypothesesDiagnostiques: string;
  planTherapeutique: string;

  // Missing/Flagged Fields
  missingFields: string;
  flaggedFields: string;
  requiresClinicalInput: string;
  requiredFor: string;

  // RAG Citations
  ragCitations: string;
  protocol: string;
  source: string;
  relevance: string;
  noProtocols: string;

  // RPA Sync
  syncToEhr: string;
  requestSync: string;
  syncPreview: string;
  approveSync: string;
  rejectSync: string;
  pendingVerifications: string;
  noPendingVerifications: string;
  syncApproved: string;
  syncRejected: string;
  humanVerificationRequired: string;
  verificationId: string;

  // Admin Dashboard
  adminDashboard: string;
  systemStatus: string;
  serviceStatus: string;

  // Protocol Management
  protocolManagement: string;
  ingestProtocols: string;
  searchProtocols: string;
  protocolStats: string;
  clearProtocols: string;
  protocolTitle: string;
  protocolContent: string;
  protocolSource: string;
  protocolCategory: string;
  protocolKeywords: string;
  totalProtocols: string;
  byLanguage: string;
  byCategory: string;
  ingestSuccess: string;
  clearConfirm: string;

  // Model Management
  modelManagement: string;
  availableModels: string;
  activeModel: string;
  selectModel: string;
  pullModel: string;
  pullingModel: string;
  modelReady: string;
  modelNotAvailable: string;
  ollamaNotRunning: string;

  // Vision Calibration
  visionCalibration: string;
  coordinateMaps: string;
  ehrType: string;
  mapName: string;
  detectLayout: string;
  saveCalibration: string;
  noCoordinateMaps: string;

  // System Services
  database: string;
  ollama: string;
  parlant: string;
  whisper: string;
  vision: string;
  qdrant: string;
  ready: string;
  notReady: string;
  notInitialized: string;

  // Errors
  errorStartSession: string;
  errorEndSession: string;
  errorLoadTranscript: string;
  errorUploadAudio: string;
  errorGenerateDraft: string;
  errorRequestSync: string;
  errorApproveSync: string;
  errorLoadSessions: string;
  errorLoadProtocols: string;
  errorLoadModels: string;
  errorLoadStatus: string;
  connectionError: string;

  // Disclaimer
  disclaimer: string;
  administrativeDisclaimer: string;
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
    save: "Save",
    cancel: "Cancel",
    delete: "Delete",
    edit: "Edit",
    search: "Search",
    loading: "Loading...",
    error: "Error",
    success: "Success",
    confirm: "Confirm",
    close: "Close",
    refresh: "Refresh",

    // Landing Page
    landingTitle: "KAIROS",
    landingSubtitle: "THE FUTURE OF ADMIN IS NOW HERE",
    receptionistPortal: "Receptionist Portal",
    clinicianPortal: "Clinician Portal",
    adminPortal: "Admin Portal",
    trustedBy: "Trusted By",
    contactUs: "Contact Us",
    getInTouch: "Get in touch with us",
    contactDescription: "Have questions about KAIROS or need help getting started? Fill out the form and our team will get back to you within 1 business day.",
    email: "Email",
    phone: "Phone",
    address: "Address",
    name: "Name",
    yourName: "Your name",
    yourEmail: "your@email.com",
    message: "Message",
    howCanWeHelp: "How can we help you?",
    submit: "Submit",

    // Authentication
    staffLogin: "Staff Login",
    enterPin: "Enter PIN",
    login: "Login",
    logout: "Logout",
    invalidPin: "Invalid PIN",

    // Receptionist Dashboard
    receptionistDashboard: "Receptionist Dashboard",
    newSession: "New Session",
    startSession: "Start Session",
    endSession: "End Session",
    pauseSession: "Pause",
    resumeSession: "Resume",
    sessionActive: "Active",
    sessionPaused: "Paused",
    sessionEnded: "Ended",
    sessionId: "Session ID",
    patientId: "Patient ID",
    language: "Language",
    selectLanguage: "Select Language",
    french: "French",
    english: "English",

    // Audio
    audioRecording: "Audio Recording",
    startRecording: "Start Recording",
    stopRecording: "Stop Recording",
    uploadAudio: "Upload Audio",
    dragDropAudio: "Drag and drop audio file here, or click to select",
    supportedFormats: "Supported formats: WAV, MP3, M4A",
    transcribing: "Transcribing...",
    transcriptEmpty: "No transcript yet. Start recording or upload an audio file.",

    // Transcript
    liveTranscript: "Live Transcript",
    transcript: "Transcript",
    confidence: "Confidence",
    keywords: "Keywords",
    noKeywords: "No keywords detected",
    speaker: "Speaker",
    timestamp: "Timestamp",

    // Clinician Dashboard
    clinicianDashboard: "Clinician Dashboard",
    sessions: "Sessions",
    selectSession: "Select a session to view details",
    noSessions: "No sessions found",
    sessionDetails: "Session Details",
    generateDraft: "Generate Draft",
    generatingDraft: "Generating draft...",

    // Compte Rendu
    compteRendu: "Compte Rendu",
    compteRenduDraft: "Compte Rendu Draft",
    patientName: "Patient Name",
    patientDob: "Date of Birth",
    patientMrn: "MRN",
    motifConsultation: "Reason for Consultation",
    anamnese: "Medical History",
    antecedents: "Past Medical History",
    allergies: "Allergies",
    traitementsActuels: "Current Treatments",
    examenClinique: "Clinical Examination",
    examensComplementaires: "Additional Tests",
    hypothesesDiagnostiques: "Diagnostic Hypotheses",
    planTherapeutique: "Treatment Plan",

    // Missing/Flagged Fields
    missingFields: "Missing Fields",
    flaggedFields: "Flagged Fields",
    requiresClinicalInput: "Requires clinical input",
    requiredFor: "Required for",

    // RAG Citations
    ragCitations: "Protocol Citations",
    protocol: "Protocol",
    source: "Source",
    relevance: "Relevance",
    noProtocols: "No protocols referenced",

    // RPA Sync
    syncToEhr: "Sync to EHR",
    requestSync: "Request Sync",
    syncPreview: "Sync Preview",
    approveSync: "Approve",
    rejectSync: "Reject",
    pendingVerifications: "Pending Verifications",
    noPendingVerifications: "No pending verifications",
    syncApproved: "Sync approved and executed",
    syncRejected: "Sync rejected",
    humanVerificationRequired: "Human verification required before syncing",
    verificationId: "Verification ID",

    // Admin Dashboard
    adminDashboard: "Admin Dashboard",
    systemStatus: "System Status",
    serviceStatus: "Service Status",

    // Protocol Management
    protocolManagement: "Protocol Management",
    ingestProtocols: "Ingest Protocols",
    searchProtocols: "Search Protocols",
    protocolStats: "Protocol Statistics",
    clearProtocols: "Clear All Protocols",
    protocolTitle: "Title",
    protocolContent: "Content",
    protocolSource: "Source",
    protocolCategory: "Category",
    protocolKeywords: "Keywords",
    totalProtocols: "Total Protocols",
    byLanguage: "By Language",
    byCategory: "By Category",
    ingestSuccess: "Protocols ingested successfully",
    clearConfirm: "Are you sure you want to clear all protocols? This cannot be undone.",

    // Model Management
    modelManagement: "Model Management",
    availableModels: "Available Models",
    activeModel: "Active Model",
    selectModel: "Select Model",
    pullModel: "Pull Model",
    pullingModel: "Pulling model...",
    modelReady: "Ready",
    modelNotAvailable: "Not Available",
    ollamaNotRunning: "Ollama is not running",

    // Vision Calibration
    visionCalibration: "Vision Calibration",
    coordinateMaps: "Coordinate Maps",
    ehrType: "EHR Type",
    mapName: "Map Name",
    detectLayout: "Auto-Detect Layout",
    saveCalibration: "Save Calibration",
    noCoordinateMaps: "No coordinate maps saved",

    // System Services
    database: "Database",
    ollama: "Ollama",
    parlant: "Parlant",
    whisper: "Whisper",
    vision: "Vision",
    qdrant: "Qdrant",
    ready: "Ready",
    notReady: "Not Ready",
    notInitialized: "Not Initialized",

    // Errors
    errorStartSession: "Failed to start session",
    errorEndSession: "Failed to end session",
    errorLoadTranscript: "Failed to load transcript",
    errorUploadAudio: "Failed to upload audio",
    errorGenerateDraft: "Failed to generate draft",
    errorRequestSync: "Failed to request sync",
    errorApproveSync: "Failed to approve sync",
    errorLoadSessions: "Failed to load sessions",
    errorLoadProtocols: "Failed to load protocols",
    errorLoadModels: "Failed to load models",
    errorLoadStatus: "Failed to load system status",
    connectionError: "Connection error. Is the backend running?",

    // Disclaimer
    disclaimer: "This tool does not provide medical diagnosis or treatment. A healthcare professional must review all outputs.",
    administrativeDisclaimer: "I am an administrative assistant. All clinical decisions require professional review.",
  },

  fr: {
    // Common
    backToHome: "Retour \u00e0 l'accueil",
    continue: "Continuer",
    processing: "Traitement en cours...",
    yes: "Oui",
    no: "Non",
    print: "Imprimer",
    returnHome: "Retour \u00e0 l'accueil",
    save: "Enregistrer",
    cancel: "Annuler",
    delete: "Supprimer",
    edit: "Modifier",
    search: "Rechercher",
    loading: "Chargement...",
    error: "Erreur",
    success: "Succ\u00e8s",
    confirm: "Confirmer",
    close: "Fermer",
    refresh: "Actualiser",

    // Landing Page
    landingTitle: "KAIROS",
    landingSubtitle: "L'AVENIR DE L'ADMINISTRATION EST ARRIV\u00c9",
    receptionistPortal: "Portail R\u00e9ception",
    clinicianPortal: "Portail Clinicien",
    adminPortal: "Portail Admin",
    trustedBy: "Ils nous font confiance",
    contactUs: "Contactez-nous",
    getInTouch: "Prenez contact avec nous",
    contactDescription: "Vous avez des questions sur KAIROS ou besoin d'aide pour commencer ? Remplissez le formulaire et notre \u00e9quipe vous r\u00e9pondra dans un d\u00e9lai d'un jour ouvrable.",
    email: "E-mail",
    phone: "T\u00e9l\u00e9phone",
    address: "Adresse",
    name: "Nom",
    yourName: "Votre nom",
    yourEmail: "votre@email.com",
    message: "Message",
    howCanWeHelp: "Comment pouvons-nous vous aider ?",
    submit: "Envoyer",

    // Authentication
    staffLogin: "Connexion Personnel",
    enterPin: "Entrez le code PIN",
    login: "Connexion",
    logout: "D\u00e9connexion",
    invalidPin: "Code PIN invalide",

    // Receptionist Dashboard
    receptionistDashboard: "Tableau de Bord R\u00e9ception",
    newSession: "Nouvelle Session",
    startSession: "D\u00e9marrer la Session",
    endSession: "Terminer la Session",
    pauseSession: "Pause",
    resumeSession: "Reprendre",
    sessionActive: "Active",
    sessionPaused: "En pause",
    sessionEnded: "Termin\u00e9e",
    sessionId: "ID Session",
    patientId: "ID Patient",
    language: "Langue",
    selectLanguage: "S\u00e9lectionner la langue",
    french: "Fran\u00e7ais",
    english: "Anglais",

    // Audio
    audioRecording: "Enregistrement Audio",
    startRecording: "D\u00e9marrer l'enregistrement",
    stopRecording: "Arr\u00eater l'enregistrement",
    uploadAudio: "T\u00e9l\u00e9charger l'audio",
    dragDropAudio: "Glissez-d\u00e9posez un fichier audio ici, ou cliquez pour s\u00e9lectionner",
    supportedFormats: "Formats support\u00e9s : WAV, MP3, M4A",
    transcribing: "Transcription en cours...",
    transcriptEmpty: "Pas encore de transcription. Commencez l'enregistrement ou t\u00e9l\u00e9chargez un fichier audio.",

    // Transcript
    liveTranscript: "Transcription en Direct",
    transcript: "Transcription",
    confidence: "Confiance",
    keywords: "Mots-cl\u00e9s",
    noKeywords: "Aucun mot-cl\u00e9 d\u00e9tect\u00e9",
    speaker: "Locuteur",
    timestamp: "Horodatage",

    // Clinician Dashboard
    clinicianDashboard: "Tableau de Bord Clinicien",
    sessions: "Sessions",
    selectSession: "S\u00e9lectionnez une session pour voir les d\u00e9tails",
    noSessions: "Aucune session trouv\u00e9e",
    sessionDetails: "D\u00e9tails de la Session",
    generateDraft: "G\u00e9n\u00e9rer le Brouillon",
    generatingDraft: "G\u00e9n\u00e9ration du brouillon...",

    // Compte Rendu
    compteRendu: "Compte Rendu",
    compteRenduDraft: "Brouillon du Compte Rendu",
    patientName: "Nom du Patient",
    patientDob: "Date de Naissance",
    patientMrn: "Num\u00e9ro de Dossier",
    motifConsultation: "Motif de Consultation",
    anamnese: "Anamn\u00e8se",
    antecedents: "Ant\u00e9c\u00e9dents",
    allergies: "Allergies",
    traitementsActuels: "Traitements Actuels",
    examenClinique: "Examen Clinique",
    examensComplementaires: "Examens Compl\u00e9mentaires",
    hypothesesDiagnostiques: "Hypoth\u00e8ses Diagnostiques",
    planTherapeutique: "Plan Th\u00e9rapeutique",

    // Missing/Flagged Fields
    missingFields: "Champs Manquants",
    flaggedFields: "Champs Signal\u00e9s",
    requiresClinicalInput: "N\u00e9cessite une saisie clinique",
    requiredFor: "Requis pour",

    // RAG Citations
    ragCitations: "Citations de Protocoles",
    protocol: "Protocole",
    source: "Source",
    relevance: "Pertinence",
    noProtocols: "Aucun protocole r\u00e9f\u00e9renc\u00e9",

    // RPA Sync
    syncToEhr: "Synchroniser avec le DPI",
    requestSync: "Demander la Synchronisation",
    syncPreview: "Aper\u00e7u de la Synchronisation",
    approveSync: "Approuver",
    rejectSync: "Rejeter",
    pendingVerifications: "V\u00e9rifications en Attente",
    noPendingVerifications: "Aucune v\u00e9rification en attente",
    syncApproved: "Synchronisation approuv\u00e9e et ex\u00e9cut\u00e9e",
    syncRejected: "Synchronisation rejet\u00e9e",
    humanVerificationRequired: "V\u00e9rification humaine requise avant synchronisation",
    verificationId: "ID de V\u00e9rification",

    // Admin Dashboard
    adminDashboard: "Tableau de Bord Admin",
    systemStatus: "\u00c9tat du Syst\u00e8me",
    serviceStatus: "\u00c9tat des Services",

    // Protocol Management
    protocolManagement: "Gestion des Protocoles",
    ingestProtocols: "Ingestion de Protocoles",
    searchProtocols: "Rechercher des Protocoles",
    protocolStats: "Statistiques des Protocoles",
    clearProtocols: "Supprimer Tous les Protocoles",
    protocolTitle: "Titre",
    protocolContent: "Contenu",
    protocolSource: "Source",
    protocolCategory: "Cat\u00e9gorie",
    protocolKeywords: "Mots-cl\u00e9s",
    totalProtocols: "Total des Protocoles",
    byLanguage: "Par Langue",
    byCategory: "Par Cat\u00e9gorie",
    ingestSuccess: "Protocoles ingestion\u00e9s avec succ\u00e8s",
    clearConfirm: "\u00cates-vous s\u00fbr de vouloir supprimer tous les protocoles ? Cette action est irr\u00e9versible.",

    // Model Management
    modelManagement: "Gestion des Mod\u00e8les",
    availableModels: "Mod\u00e8les Disponibles",
    activeModel: "Mod\u00e8le Actif",
    selectModel: "S\u00e9lectionner le Mod\u00e8le",
    pullModel: "T\u00e9l\u00e9charger le Mod\u00e8le",
    pullingModel: "T\u00e9l\u00e9chargement du mod\u00e8le...",
    modelReady: "Pr\u00eat",
    modelNotAvailable: "Non Disponible",
    ollamaNotRunning: "Ollama n'est pas en cours d'ex\u00e9cution",

    // Vision Calibration
    visionCalibration: "Calibration Vision",
    coordinateMaps: "Cartes de Coordonn\u00e9es",
    ehrType: "Type de DPI",
    mapName: "Nom de la Carte",
    detectLayout: "D\u00e9tection Automatique",
    saveCalibration: "Enregistrer la Calibration",
    noCoordinateMaps: "Aucune carte de coordonn\u00e9es enregistr\u00e9e",

    // System Services
    database: "Base de donn\u00e9es",
    ollama: "Ollama",
    parlant: "Parlant",
    whisper: "Whisper",
    vision: "Vision",
    qdrant: "Qdrant",
    ready: "Pr\u00eat",
    notReady: "Non Pr\u00eat",
    notInitialized: "Non Initialis\u00e9",

    // Errors
    errorStartSession: "\u00c9chec du d\u00e9marrage de la session",
    errorEndSession: "\u00c9chec de la fin de session",
    errorLoadTranscript: "\u00c9chec du chargement de la transcription",
    errorUploadAudio: "\u00c9chec du t\u00e9l\u00e9chargement audio",
    errorGenerateDraft: "\u00c9chec de la g\u00e9n\u00e9ration du brouillon",
    errorRequestSync: "\u00c9chec de la demande de synchronisation",
    errorApproveSync: "\u00c9chec de l'approbation de la synchronisation",
    errorLoadSessions: "\u00c9chec du chargement des sessions",
    errorLoadProtocols: "\u00c9chec du chargement des protocoles",
    errorLoadModels: "\u00c9chec du chargement des mod\u00e8les",
    errorLoadStatus: "\u00c9chec du chargement de l'\u00e9tat du syst\u00e8me",
    connectionError: "Erreur de connexion. Le serveur est-il en marche ?",

    // Disclaimer
    disclaimer: "Cet outil ne fournit pas de diagnostic ou de traitement m\u00e9dical. Un professionnel de sant\u00e9 doit examiner tous les r\u00e9sultats.",
    administrativeDisclaimer: "Je suis un assistant administratif. Toutes les d\u00e9cisions cliniques n\u00e9cessitent un examen professionnel.",
  },
};

// Helper function to get translation
export function t(key: keyof Translations, language: LanguageCode = "en"): string {
  return translations[language][key] || translations.en[key] || key;
}

// Hook-friendly helper
export function useTranslation(language: LanguageCode) {
  return {
    t: (key: keyof Translations) => t(key, language),
    language,
  };
}
