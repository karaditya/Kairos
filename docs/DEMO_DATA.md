# KAIROS CAE - Demo Patient Data

This document contains example data showing the output at each stage of the clinical documentation workflow.

---

## Demo Patient 1: Acute Respiratory Complaint

### Stage 1: Raw Audio Transcription Output

**Session Metadata:**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "patient_id": "PAT-2024-0142",
  "language": "fr",
  "status": "completed",
  "created_at": "2024-01-15T09:30:00Z",
  "duration_seconds": 423.5
}
```

**Transcript Segments:**
```json
{
  "transcript_id": "tr-001-2024",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "full_text": "Bonjour Monsieur Dupont, comment allez-vous aujourd'hui? Pas très bien docteur, j'ai une toux depuis trois jours qui ne passe pas. Je tousse beaucoup la nuit et j'ai du mal à dormir. D'accord, avez-vous de la fièvre? Oui, j'ai pris ma température ce matin, j'avais 38.5 degrés. Est-ce que vous avez des difficultés à respirer? Un peu oui, surtout quand je monte les escaliers. Je suis essoufflé. Avez-vous des antécédents médicaux? Je suis diabétique de type 2 depuis cinq ans. Je prends du Metformine. Des allergies connues? Oui, je suis allergique à la pénicilline. Très bien, je vais vous examiner. Respirez profondément s'il vous plaît. J'entends des râles à la base du poumon droit. Je vais vous prescrire une radiographie thoracique et un bilan sanguin. En attendant, je vous prescris du paracétamol pour la fièvre et un sirop antitussif. Revenez me voir dans trois jours avec les résultats.",
  "segments": [
    {
      "id": 1,
      "text": "Bonjour Monsieur Dupont, comment allez-vous aujourd'hui?",
      "start_time": 0.0,
      "end_time": 3.2,
      "confidence": 0.97,
      "speaker": "clinician",
      "keywords": []
    },
    {
      "id": 2,
      "text": "Pas très bien docteur, j'ai une toux depuis trois jours qui ne passe pas.",
      "start_time": 3.5,
      "end_time": 8.1,
      "confidence": 0.95,
      "speaker": "patient",
      "keywords": ["toux", "trois jours"]
    },
    {
      "id": 3,
      "text": "Je tousse beaucoup la nuit et j'ai du mal à dormir.",
      "start_time": 8.3,
      "end_time": 12.0,
      "confidence": 0.94,
      "speaker": "patient",
      "keywords": ["tousse", "nuit", "dormir"]
    },
    {
      "id": 4,
      "text": "D'accord, avez-vous de la fièvre?",
      "start_time": 12.5,
      "end_time": 14.2,
      "confidence": 0.98,
      "speaker": "clinician",
      "keywords": ["fièvre"]
    },
    {
      "id": 5,
      "text": "Oui, j'ai pris ma température ce matin, j'avais 38.5 degrés.",
      "start_time": 14.5,
      "end_time": 18.8,
      "confidence": 0.96,
      "speaker": "patient",
      "keywords": ["température", "38.5", "fièvre"]
    },
    {
      "id": 6,
      "text": "Est-ce que vous avez des difficultés à respirer?",
      "start_time": 19.2,
      "end_time": 22.0,
      "confidence": 0.97,
      "speaker": "clinician",
      "keywords": ["difficultés", "respirer"]
    },
    {
      "id": 7,
      "text": "Un peu oui, surtout quand je monte les escaliers. Je suis essoufflé.",
      "start_time": 22.3,
      "end_time": 27.5,
      "confidence": 0.93,
      "speaker": "patient",
      "keywords": ["essoufflé", "escaliers", "dyspnée"]
    },
    {
      "id": 8,
      "text": "Avez-vous des antécédents médicaux?",
      "start_time": 28.0,
      "end_time": 30.2,
      "confidence": 0.98,
      "speaker": "clinician",
      "keywords": ["antécédents"]
    },
    {
      "id": 9,
      "text": "Je suis diabétique de type 2 depuis cinq ans. Je prends du Metformine.",
      "start_time": 30.5,
      "end_time": 36.0,
      "confidence": 0.95,
      "speaker": "patient",
      "keywords": ["diabétique", "type 2", "Metformine"]
    },
    {
      "id": 10,
      "text": "Des allergies connues?",
      "start_time": 36.5,
      "end_time": 38.0,
      "confidence": 0.99,
      "speaker": "clinician",
      "keywords": ["allergies"]
    },
    {
      "id": 11,
      "text": "Oui, je suis allergique à la pénicilline.",
      "start_time": 38.3,
      "end_time": 41.0,
      "confidence": 0.97,
      "speaker": "patient",
      "keywords": ["allergique", "pénicilline"]
    },
    {
      "id": 12,
      "text": "Très bien, je vais vous examiner. Respirez profondément s'il vous plaît.",
      "start_time": 42.0,
      "end_time": 46.5,
      "confidence": 0.96,
      "speaker": "clinician",
      "keywords": ["examiner", "respirez"]
    },
    {
      "id": 13,
      "text": "J'entends des râles à la base du poumon droit.",
      "start_time": 55.0,
      "end_time": 58.5,
      "confidence": 0.94,
      "speaker": "clinician",
      "keywords": ["râles", "poumon droit", "auscultation"]
    },
    {
      "id": 14,
      "text": "Je vais vous prescrire une radiographie thoracique et un bilan sanguin.",
      "start_time": 59.0,
      "end_time": 64.0,
      "confidence": 0.96,
      "speaker": "clinician",
      "keywords": ["radiographie", "thoracique", "bilan sanguin"]
    },
    {
      "id": 15,
      "text": "En attendant, je vous prescris du paracétamol pour la fièvre et un sirop antitussif.",
      "start_time": 64.5,
      "end_time": 70.0,
      "confidence": 0.95,
      "speaker": "clinician",
      "keywords": ["paracétamol", "fièvre", "antitussif"]
    },
    {
      "id": 16,
      "text": "Revenez me voir dans trois jours avec les résultats.",
      "start_time": 70.5,
      "end_time": 74.0,
      "confidence": 0.97,
      "speaker": "clinician",
      "keywords": ["trois jours", "résultats", "suivi"]
    }
  ],
  "keywords": [
    "toux",
    "fièvre",
    "38.5",
    "essoufflé",
    "dyspnée",
    "diabétique",
    "type 2",
    "Metformine",
    "pénicilline",
    "allergie",
    "râles",
    "poumon",
    "radiographie",
    "paracétamol",
    "antitussif"
  ],
  "keywords_by_category": {
    "symptoms": ["toux", "fièvre", "essoufflé", "dyspnée"],
    "vitals": ["38.5"],
    "conditions": ["diabétique", "type 2"],
    "medications": ["Metformine", "paracétamol", "antitussif"],
    "allergies": ["pénicilline"],
    "examinations": ["râles", "poumon", "auscultation"],
    "tests": ["radiographie", "bilan sanguin"]
  }
}
```

---

### Stage 2: Generated Compte Rendu

**API Response from `GET /agent/draft?session_id=a1b2c3d4-e5f6-7890-abcd-ef1234567890`:**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "compte_rendu_id": "cr-001-2024",
  "compte_rendu": {
    "patient_name": "M. Dupont",
    "patient_dob": null,
    "patient_mrn": "PAT-2024-0142",
    "consultation_date": "2024-01-15",

    "motif_consultation": "Toux persistante depuis 3 jours avec fièvre",

    "anamnese": "Patient de sexe masculin se présentant pour une toux évoluant depuis trois jours, à prédominance nocturne, perturbant le sommeil. Fièvre associée mesurée à 38.5°C ce matin. Dyspnée d'effort rapportée, notamment à la montée des escaliers.",

    "antecedents": {
      "medicaux": ["Diabète de type 2 (depuis 5 ans)"],
      "chirurgicaux": [],
      "familiaux": []
    },

    "allergies": ["Pénicilline"],

    "traitements_actuels": [
      {
        "medication": "Metformine",
        "dosage": "Non précisé",
        "frequency": "Non précisé"
      }
    ],

    "examen_clinique": {
      "general": "Patient fébrile",
      "respiratoire": "Auscultation pulmonaire: râles crépitants à la base du poumon droit",
      "cardiovasculaire": null,
      "abdominal": null,
      "neurologique": null,
      "autres": null
    },

    "examens_complementaires": {
      "prescrits": [
        "Radiographie thoracique",
        "Bilan sanguin (NFS, CRP)"
      ],
      "resultats": null
    },

    "hypotheses_diagnostiques": [
      {
        "diagnostic": "Pneumopathie infectieuse de la base droite",
        "probability": "élevée",
        "justification": "Toux + fièvre + râles crépitants localisés"
      },
      {
        "diagnostic": "Bronchite aiguë",
        "probability": "modérée",
        "justification": "Tableau clinique compatible mais râles localisés orientent vers pneumopathie"
      }
    ],

    "plan_therapeutique": {
      "prescriptions": [
        {
          "medication": "Paracétamol",
          "dosage": "1g",
          "frequency": "Toutes les 6 heures si fièvre",
          "duration": "Selon besoin"
        },
        {
          "medication": "Sirop antitussif",
          "dosage": "Selon notice",
          "frequency": "3 fois par jour",
          "duration": "5 jours"
        }
      ],
      "consignes": [
        "Repos",
        "Hydratation abondante",
        "Surveillance température"
      ],
      "suivi": "Consultation de contrôle dans 3 jours avec résultats des examens"
    }
  },

  "missing_fields": [
    {
      "field_name": "patient_dob",
      "reason": "Date de naissance non mentionnée dans la consultation",
      "required_for": "Identification patient"
    },
    {
      "field_name": "traitements_actuels.dosage",
      "reason": "Posologie du Metformine non précisée",
      "required_for": "Bilan médicamenteux complet"
    }
  ],

  "flagged_fields": [
    {
      "field_name": "hypotheses_diagnostiques",
      "reason": "Diagnostic de pneumopathie nécessite confirmation radiologique",
      "requires_clinical_input": true
    },
    {
      "field_name": "plan_therapeutique.prescriptions",
      "reason": "Vérifier si antibiothérapie nécessaire après résultats (attention allergie pénicilline)",
      "requires_clinical_input": true
    }
  ],

  "rag_citations": [
    {
      "protocol": "Protocole de prise en charge des pneumopathies communautaires",
      "source": "HAS Guidelines 2023",
      "relevance": 0.92,
      "excerpt": "Devant un tableau associant toux, fièvre et anomalies auscultatoires localisées, une radiographie thoracique est recommandée..."
    },
    {
      "protocol": "Antibiothérapie et allergie aux bêta-lactamines",
      "source": "ANSM Recommandations",
      "relevance": 0.87,
      "excerpt": "En cas d'allergie documentée à la pénicilline, privilégier les macrolides ou fluoroquinolones..."
    }
  ],

  "model_used": "Parlant (Scribe Agent) + mistral:7b-instruct",
  "generated_at": "2024-01-15T09:45:23Z",
  "disclaimer": "Ce document a été généré par un assistant administratif IA. Toutes les informations cliniques doivent être vérifiées et validées par le médecin traitant."
}
```

---

### Stage 3: EHR Vision Capture Output

**API Response from `POST /vision/scrape`:**

```json
{
  "success": true,
  "screenshot_id": "scr-001-2024",
  "extracted_data": {
    "patient_name": {
      "value": "DUPONT Jean-Pierre",
      "confidence": 0.95,
      "source": "vision_ocr",
      "bbox": [120, 85, 280, 105]
    },
    "patient_dob": {
      "value": "15/03/1965",
      "confidence": 0.93,
      "source": "vision_ocr",
      "bbox": [320, 85, 420, 105]
    },
    "patient_mrn": {
      "value": "PAT-2024-0142",
      "confidence": 0.98,
      "source": "vision_ocr",
      "bbox": [450, 85, 580, 105]
    },
    "current_notes": {
      "value": "",
      "confidence": 1.0,
      "source": "vision_ocr",
      "bbox": [50, 200, 750, 500]
    },
    "allergies_field": {
      "value": "Pénicilline",
      "confidence": 0.91,
      "source": "vision_ocr",
      "bbox": [600, 120, 750, 140]
    }
  },
  "ui_elements": [
    {
      "type": "input_field",
      "label": "Notes cliniques",
      "bbox": [50, 200, 750, 500],
      "editable": true
    },
    {
      "type": "button",
      "label": "Enregistrer",
      "bbox": [650, 520, 750, 550],
      "editable": false
    },
    {
      "type": "input_field",
      "label": "Prescriptions",
      "bbox": [50, 560, 750, 700],
      "editable": true
    }
  ],
  "raw_text": "Dossier Patient - Clinique Saint-Martin\nNom: DUPONT Jean-Pierre  DOB: 15/03/1965  MRN: PAT-2024-0142\nAllergies: Pénicilline\n\nNotes cliniques:\n[Zone de saisie vide]\n\nPrescriptions:\n[Zone de saisie vide]"
}
```

---

### Stage 4: RPA Sync Preview

**API Response from `POST /agent/sync`:**

```json
{
  "status": "pending_approval",
  "verification_id": "ver-001-2024",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "compte_rendu_id": "cr-001-2024",
  "created_at": "2024-01-15T09:50:00Z",

  "preview": {
    "target_window": "Dossier Patient - Clinique Saint-Martin",
    "total_actions": 8,
    "estimated_duration_seconds": 15,

    "actions": [
      {
        "sequence": 1,
        "type": "CLICK",
        "target_field": "notes_cliniques",
        "coordinates": {"x": 400, "y": 350},
        "description": "Cliquer sur le champ Notes cliniques"
      },
      {
        "sequence": 2,
        "type": "CLEAR",
        "target_field": "notes_cliniques",
        "description": "Effacer le contenu existant (Ctrl+A, Delete)"
      },
      {
        "sequence": 3,
        "type": "TYPE",
        "target_field": "notes_cliniques",
        "value": "CONSULTATION DU 15/01/2024\n\nMotif: Toux persistante depuis 3 jours avec fièvre\n\nAnamnèse: Patient présentant toux à prédominance nocturne, fièvre 38.5°C, dyspnée d'effort.\n\nAntécédents: Diabète type 2 (5 ans) - Metformine\nAllergies: PÉNICILLINE\n\nExamen: Râles crépitants base poumon droit\n\nCAT: Radio thorax + bilan sanguin prescrits\nTraitement: Paracétamol 1g/6h + sirop antitussif\nSuivi: RDV J+3 avec résultats",
        "description": "Saisir le compte rendu dans Notes cliniques"
      },
      {
        "sequence": 4,
        "type": "VERIFY",
        "target_field": "notes_cliniques",
        "expected_contains": "Toux persistante",
        "description": "Vérifier que le texte a été saisi correctement"
      },
      {
        "sequence": 5,
        "type": "CLICK",
        "target_field": "prescriptions",
        "coordinates": {"x": 400, "y": 630},
        "description": "Cliquer sur le champ Prescriptions"
      },
      {
        "sequence": 6,
        "type": "TYPE",
        "target_field": "prescriptions",
        "value": "1. Paracétamol 1g - 1 comprimé toutes les 6h si fièvre\n2. Sirop antitussif - 1 cuillère à soupe 3x/jour pendant 5 jours",
        "description": "Saisir les prescriptions"
      },
      {
        "sequence": 7,
        "type": "VERIFY",
        "target_field": "prescriptions",
        "expected_contains": "Paracétamol",
        "description": "Vérifier que les prescriptions ont été saisies"
      },
      {
        "sequence": 8,
        "type": "WAIT",
        "duration_ms": 500,
        "description": "Pause avant sauvegarde (optionnel)"
      }
    ],

    "warnings": [
      "Le bouton 'Enregistrer' ne sera PAS cliqué automatiquement - action manuelle requise",
      "Vérifier l'allergie PÉNICILLINE avant toute prescription d'antibiotique"
    ]
  }
}
```

---

### Stage 5: RPA Execution Result

**API Response from `POST /agent/sync/{verification_id}/approve`:**

```json
{
  "status": "completed",
  "verification_id": "ver-001-2024",
  "approved_by": "DR-MARTIN",
  "approved_at": "2024-01-15T09:52:30Z",
  "executed_at": "2024-01-15T09:52:31Z",

  "execution_result": {
    "success": true,
    "actions_executed": 8,
    "actions_successful": 8,
    "actions_failed": 0,
    "total_duration_ms": 12450,

    "action_results": [
      {
        "sequence": 1,
        "type": "CLICK",
        "success": true,
        "duration_ms": 150
      },
      {
        "sequence": 2,
        "type": "CLEAR",
        "success": true,
        "duration_ms": 200
      },
      {
        "sequence": 3,
        "type": "TYPE",
        "success": true,
        "duration_ms": 8500,
        "characters_typed": 425
      },
      {
        "sequence": 4,
        "type": "VERIFY",
        "success": true,
        "duration_ms": 1200,
        "verification": {
          "method": "vision_ocr",
          "expected": "Toux persistante",
          "found": true,
          "confidence": 0.94
        }
      },
      {
        "sequence": 5,
        "type": "CLICK",
        "success": true,
        "duration_ms": 150
      },
      {
        "sequence": 6,
        "type": "TYPE",
        "success": true,
        "duration_ms": 1800,
        "characters_typed": 142
      },
      {
        "sequence": 7,
        "type": "VERIFY",
        "success": true,
        "duration_ms": 950,
        "verification": {
          "method": "vision_ocr",
          "expected": "Paracétamol",
          "found": true,
          "confidence": 0.96
        }
      },
      {
        "sequence": 8,
        "type": "WAIT",
        "success": true,
        "duration_ms": 500
      }
    ],

    "post_sync_verification": {
      "performed": true,
      "screenshot_id": "scr-002-2024",
      "fields_verified": {
        "notes_cliniques": {
          "expected_content": "Toux persistante",
          "verified": true,
          "confidence": 0.94
        },
        "prescriptions": {
          "expected_content": "Paracétamol",
          "verified": true,
          "confidence": 0.96
        }
      },
      "discrepancies": []
    }
  },

  "audit_trail": {
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "compte_rendu_id": "cr-001-2024",
    "verification_requested_at": "2024-01-15T09:50:00Z",
    "verification_approved_at": "2024-01-15T09:52:30Z",
    "execution_completed_at": "2024-01-15T09:52:43Z",
    "approved_by": "DR-MARTIN",
    "workstation": "POSTE-CONSULTATION-01"
  }
}
```

---

## Demo Patient 2: Follow-up Consultation (English)

### Transcript Output

```json
{
  "session_id": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
  "patient_id": "PAT-2024-0089",
  "language": "en",
  "status": "completed",
  "full_text": "Good morning Mrs. Johnson, please have a seat. How are you feeling today? Much better doctor, thank you. The antibiotics really helped with my sinus infection. That's great to hear. Any remaining symptoms? Just a little bit of congestion in the morning, but it clears up quickly. Let me take a look. Your sinuses look clear now. I'd recommend continuing with saline rinses for another week. Do you have any other concerns? Actually yes, I've been having some lower back pain for about two weeks. It started after I helped my daughter move. Is it constant or does it come and go? It's worse in the morning and when I sit for too long. On a scale of 1 to 10, how would you rate the pain? About a 5 or 6. Any numbness or tingling in your legs? No, nothing like that. Good, that's reassuring. Based on your description, this sounds like muscular strain. I'll prescribe some anti-inflammatory medication and recommend gentle stretching exercises. If it doesn't improve in two weeks, we'll order an X-ray. Thank you doctor.",
  "segments": [
    {
      "id": 1,
      "text": "Good morning Mrs. Johnson, please have a seat. How are you feeling today?",
      "start_time": 0.0,
      "end_time": 4.5,
      "confidence": 0.96,
      "speaker": "clinician",
      "keywords": []
    },
    {
      "id": 2,
      "text": "Much better doctor, thank you. The antibiotics really helped with my sinus infection.",
      "start_time": 5.0,
      "end_time": 10.2,
      "confidence": 0.94,
      "speaker": "patient",
      "keywords": ["antibiotics", "sinus infection"]
    },
    {
      "id": 3,
      "text": "That's great to hear. Any remaining symptoms?",
      "start_time": 10.5,
      "end_time": 13.0,
      "confidence": 0.97,
      "speaker": "clinician",
      "keywords": ["symptoms"]
    },
    {
      "id": 4,
      "text": "Just a little bit of congestion in the morning, but it clears up quickly.",
      "start_time": 13.5,
      "end_time": 17.5,
      "confidence": 0.95,
      "speaker": "patient",
      "keywords": ["congestion", "morning"]
    },
    {
      "id": 5,
      "text": "Let me take a look. Your sinuses look clear now. I'd recommend continuing with saline rinses for another week.",
      "start_time": 18.0,
      "end_time": 25.0,
      "confidence": 0.93,
      "speaker": "clinician",
      "keywords": ["sinuses", "saline rinses"]
    },
    {
      "id": 6,
      "text": "Do you have any other concerns?",
      "start_time": 25.5,
      "end_time": 27.5,
      "confidence": 0.98,
      "speaker": "clinician",
      "keywords": []
    },
    {
      "id": 7,
      "text": "Actually yes, I've been having some lower back pain for about two weeks.",
      "start_time": 28.0,
      "end_time": 33.0,
      "confidence": 0.95,
      "speaker": "patient",
      "keywords": ["lower back pain", "two weeks"]
    },
    {
      "id": 8,
      "text": "It started after I helped my daughter move.",
      "start_time": 33.5,
      "end_time": 36.5,
      "confidence": 0.94,
      "speaker": "patient",
      "keywords": ["started", "move"]
    },
    {
      "id": 9,
      "text": "Is it constant or does it come and go?",
      "start_time": 37.0,
      "end_time": 39.5,
      "confidence": 0.97,
      "speaker": "clinician",
      "keywords": []
    },
    {
      "id": 10,
      "text": "It's worse in the morning and when I sit for too long.",
      "start_time": 40.0,
      "end_time": 44.0,
      "confidence": 0.95,
      "speaker": "patient",
      "keywords": ["morning", "sitting"]
    },
    {
      "id": 11,
      "text": "On a scale of 1 to 10, how would you rate the pain?",
      "start_time": 44.5,
      "end_time": 48.0,
      "confidence": 0.96,
      "speaker": "clinician",
      "keywords": ["pain scale"]
    },
    {
      "id": 12,
      "text": "About a 5 or 6.",
      "start_time": 48.5,
      "end_time": 50.0,
      "confidence": 0.98,
      "speaker": "patient",
      "keywords": ["5", "6", "pain"]
    },
    {
      "id": 13,
      "text": "Any numbness or tingling in your legs?",
      "start_time": 50.5,
      "end_time": 53.0,
      "confidence": 0.97,
      "speaker": "clinician",
      "keywords": ["numbness", "tingling", "legs"]
    },
    {
      "id": 14,
      "text": "No, nothing like that.",
      "start_time": 53.5,
      "end_time": 55.0,
      "confidence": 0.99,
      "speaker": "patient",
      "keywords": []
    },
    {
      "id": 15,
      "text": "Good, that's reassuring. Based on your description, this sounds like muscular strain.",
      "start_time": 55.5,
      "end_time": 61.0,
      "confidence": 0.94,
      "speaker": "clinician",
      "keywords": ["muscular strain"]
    },
    {
      "id": 16,
      "text": "I'll prescribe some anti-inflammatory medication and recommend gentle stretching exercises.",
      "start_time": 61.5,
      "end_time": 67.0,
      "confidence": 0.95,
      "speaker": "clinician",
      "keywords": ["anti-inflammatory", "stretching exercises"]
    },
    {
      "id": 17,
      "text": "If it doesn't improve in two weeks, we'll order an X-ray.",
      "start_time": 67.5,
      "end_time": 71.5,
      "confidence": 0.96,
      "speaker": "clinician",
      "keywords": ["two weeks", "X-ray"]
    },
    {
      "id": 18,
      "text": "Thank you doctor.",
      "start_time": 72.0,
      "end_time": 73.5,
      "confidence": 0.99,
      "speaker": "patient",
      "keywords": []
    }
  ],
  "keywords_by_category": {
    "symptoms": ["congestion", "lower back pain", "numbness", "tingling"],
    "conditions": ["sinus infection", "muscular strain"],
    "medications": ["antibiotics", "anti-inflammatory", "saline rinses"],
    "examinations": ["sinuses"],
    "tests": ["X-ray"],
    "vitals": ["pain scale 5-6"]
  }
}
```

---

## Demo Data Files for Testing

### File: `demo_sessions.json`

```json
[
  {
    "session_id": "demo-session-001",
    "patient_id": "DEMO-PAT-001",
    "patient_name": "Jean-Pierre Dupont",
    "language": "fr",
    "status": "completed",
    "chief_complaint": "Toux et fièvre depuis 3 jours",
    "created_at": "2024-01-15T09:30:00Z"
  },
  {
    "session_id": "demo-session-002",
    "patient_id": "DEMO-PAT-002",
    "patient_name": "Marie Johnson",
    "language": "en",
    "status": "completed",
    "chief_complaint": "Follow-up sinus infection + new back pain",
    "created_at": "2024-01-15T10:15:00Z"
  },
  {
    "session_id": "demo-session-003",
    "patient_id": "DEMO-PAT-003",
    "patient_name": "Ahmed Benali",
    "language": "fr",
    "status": "active",
    "chief_complaint": "Douleur thoracique",
    "created_at": "2024-01-15T11:00:00Z"
  }
]
```

### File: `demo_protocols.json`

```json
[
  {
    "title": "Protocole pneumopathie communautaire",
    "content": "Devant un tableau associant toux, fièvre et anomalies auscultatoires localisées, une radiographie thoracique est recommandée. En première intention, une antibiothérapie par amoxicilline 1g x3/jour est indiquée, sauf allergie documentée aux bêta-lactamines.",
    "source": "HAS Guidelines 2023",
    "language": "fr",
    "category": "respiratory"
  },
  {
    "title": "Allergie pénicilline - alternatives",
    "content": "En cas d'allergie documentée à la pénicilline, privilégier: 1) Macrolides (azithromycine, clarithromycine) pour infections respiratoires, 2) Fluoroquinolones (lévofloxacine) en 2ème intention, 3) Eviter toutes bêta-lactamines en cas d'allergie sévère.",
    "source": "ANSM Recommandations",
    "language": "fr",
    "category": "allergology"
  },
  {
    "title": "Low back pain management protocol",
    "content": "For acute mechanical low back pain without red flags: 1) NSAIDs first-line (ibuprofen 400mg TID), 2) Muscle relaxants if spasm present, 3) Activity modification, avoid bed rest, 4) Physical therapy referral if not improving in 2-4 weeks, 5) Imaging only if red flags or no improvement at 6 weeks.",
    "source": "NICE Guidelines 2023",
    "language": "en",
    "category": "musculoskeletal"
  }
]
```

---

## Seeding Demo Data

To load this demo data into the system:

```bash
# Load demo protocols
curl -X POST http://localhost:8000/admin/protocols/ingest \
  -H "Content-Type: application/json" \
  -H "X-Staff-PIN: 1234" \
  -d @demo_protocols.json

# Create demo session
curl -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "DEMO-PAT-001", "language": "fr"}'
```
