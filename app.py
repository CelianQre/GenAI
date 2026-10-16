from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import streamlit as st
from openai import OpenAI


def load_local_environment() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as env_file:
        for line in env_file:
            entry = line.strip()
            if not entry or entry.startswith("#") or "=" not in entry:
                continue
            name, value = entry.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


load_local_environment()

st.set_page_config(
    page_title="Programmateur",
    layout="wide",
)


@dataclass
class GeneratedResult:
    code: str
    tests: str
    explanation: str
    analysis: str
    candidates: list[str]
    selected_approach: str
    react_steps: list[tuple[str, str]]
    correction_applied: bool


def extract_json_payload(raw_content: str) -> dict:
    content = raw_content.strip()
    decoder = json.JSONDecoder()
    for start, character in enumerate(content):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(content[start:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("La réponse Ollama ne contient pas de JSON valide.")


def normalize_python_block(content: object) -> str:
    if not isinstance(content, str):
        return ""

    normalized = content.strip()
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", normalized, re.DOTALL | re.IGNORECASE)
    if fenced:
        normalized = fenced.group(1).strip()
    elif normalized.lower().startswith("python\n"):
        normalized = normalized[7:].lstrip()
    return normalized


def normalize_payload(payload: dict) -> dict:
    for field in ("code", "tests"):
        payload[field] = normalize_python_block(payload.get(field))
    if isinstance(payload.get("candidates"), str):
        payload["candidates"] = [
            candidate.strip(" -")
            for candidate in re.split(r"\r?\n|;", payload["candidates"])
            if candidate.strip(" -")
        ]
    return payload


def validate_reasoning_fields(payload: dict) -> list[str]:
    errors = []
    if not isinstance(payload.get("analysis"), str) or not payload["analysis"].strip():
        errors.append("analysis doit contenir une analyse structurée")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2 or not all(isinstance(item, str) for item in candidates):
        errors.append("candidates doit contenir au moins deux approches")
    if not isinstance(payload.get("selected_approach"), str) or not payload["selected_approach"].strip():
        errors.append("selected_approach doit contenir l'approche retenue")
    return errors


def validate_python(code: str, tests: str) -> list[str]:
    errors = []
    for filename, content in (("solution.py", code), ("test_solution.py", tests)):
        try:
            ast.parse(content, filename=filename)
        except SyntaxError as error:
            errors.append(f"{filename}: ligne {error.lineno}, {error.msg}")
    return errors


def validate_pytest_file(tests: str) -> list[str]:
    errors = []
    try:
        tree = ast.parse(tests, filename="test_solution.py")
    except SyntaxError:
        return errors

    has_pytest_import = any(
        isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names)
        or isinstance(node, ast.ImportFrom) and node.module == "pytest"
        for node in tree.body
    )
    if not has_pytest_import:
        errors.append("test_solution.py doit contenir 'import pytest'")

    has_solution_import = any(
        isinstance(node, ast.ImportFrom) and node.module == "solution"
        or isinstance(node, ast.Import) and any(alias.name == "solution" for alias in node.names)
        for node in tree.body
    )
    if not has_solution_import:
        errors.append("test_solution.py doit importer solution.py")

    has_test_function = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    if not has_test_function:
        errors.append("test_solution.py doit contenir au moins une fonction nommée 'test_*'")

    return errors


def run_pytest_tool(code: str, tests: str) -> tuple[bool, str]:
    """Exécute les tests générés dans un dossier temporaire isolé du projet."""
    python_candidates = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"),
    ]
    pytest_python = sys.executable
    for candidate in python_candidates:
        if not os.path.exists(candidate):
            continue
        check_pytest = subprocess.run(
            [candidate, "-c", "import pytest"],
            capture_output=True,
            text=True,
            check=False,
        )
        if check_pytest.returncode == 0:
            pytest_python = candidate
            break

    with tempfile.TemporaryDirectory(prefix="programmateur_") as temp_dir:
        solution_path = os.path.join(temp_dir, "solution.py")
        tests_path = os.path.join(temp_dir, "test_solution.py")
        with open(solution_path, "w", encoding="utf-8") as solution_file:
            solution_file.write(code)
        with open(tests_path, "w", encoding="utf-8") as tests_file:
            tests_file.write(tests)

        try:
            completed = subprocess.run(
                [pytest_python, "-m", "pytest", "-q", "test_solution.py"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "pytest interrompu après 20 secondes."

        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        return completed.returncode == 0, output[-4000:] or "pytest n'a produit aucune sortie."


def get_ollama_client() -> tuple[OpenAI, str, int]:
    client = OpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        timeout=float(os.getenv("OLLAMA_TIMEOUT", "90")),
    )
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    max_tokens = max(int(os.getenv("OLLAMA_MAX_TOKENS", "3600")), 3600)
    return client, model, max_tokens


def ollama_generation(request: str) -> GeneratedResult:
    client, model, max_tokens = get_ollama_client()
    required_fields = {
        "analysis",
        "candidates",
        "selected_approach",
        "code",
        "tests",
        "explanation",
    }

    prompt = f"""Tu es un développeur senior et tu utilises une méthode d'agent explicite.
Avant de répondre, applique silencieusement les étapes suivantes :
1. Chain of Thought structurée : décomposer la demande en contraintes, entrées, sorties et cas limites.
2. Tree of Thoughts : proposer au moins deux approches possibles, les comparer selon la simplicité,
   la robustesse et la facilité de test, puis retenir la meilleure.
3. ReAct : transformer l'approche retenue en action (code et tests), puis observer mentalement les
   imports, les cas limites et la cohérence entre solution.py et test_solution.py.
Ne révèle jamais cette réflexion détaillée dans la réponse. Retourne uniquement l'objet JSON demandé.

Réponds uniquement avec un objet JSON valide au format exact suivant :
{{"analysis": "...", "candidates": ["...", "..."], "selected_approach": "...", "code": "...", "tests": "...", "explanation": "..."}}

Spécifications :
- Demande : {request}
- Le champ "code" contient l'implémentation Python complète (solution.py).
- Le champ "tests" contient le fichier complet test_solution.py qui importe pytest et les fonctions depuis solution.py.
- tests doit contenir au moins une fonction de module nommée `test_*`, par exemple `def test_resultat():`.
- tests doit importer les fonctions depuis `solution.py`, par exemple `from solution import fonction`.
- Le champ "analysis" résume les contraintes, les entrées, les sorties et les cas limites.
- Le champ "candidates" contient au moins deux approches distinctes envisagées.
- Le champ "selected_approach" explique brièvement pourquoi une approche a été retenue.
- Le champ "explanation" résume brièvement l'approche.
- Les tests doivent rester courts et utiliser principalement des appels et des assertions simples.
- N'utilise aucune chaîne multilignes et termine chaque chaîne Python avec ses guillemets fermants.
- Tout le code doit être sous forme de chaîne de caractères échappée, jamais sous forme de tableaux ou d'objets."""

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Tu réponds exclusivement en JSON valide sans fioritures."},
            {"role": "user", "content": prompt},
        ],
    )

    payload = normalize_payload(extract_json_payload(response.choices[0].message.content))
    react_steps = [("Analyse", payload.get("analysis", "Analyse produite par le modèle."))]
    missing_fields = required_fields - payload.keys()
    errors = [f"Champs JSON manquants : {', '.join(sorted(missing_fields))}"] if missing_fields else validate_reasoning_fields(payload)
    if not errors:
        react_steps.append(("Tree of Thoughts", "Pistes envisagées :\n- " + "\n- ".join(payload["candidates"]) + "\n\nApproche retenue : " + payload["selected_approach"]))
        errors = validate_python(payload["code"], payload["tests"]) + validate_pytest_file(payload["tests"])
        react_steps.append(("Action", "Exécuter l'outil pytest dans un dossier temporaire."))
        if not errors:
            tests_passed, observation = run_pytest_tool(payload["code"], payload["tests"])
            react_steps.append(("Observation", observation))
            if not tests_passed:
                errors = ["Échec de pytest :\n" + observation]
        else:
            react_steps.append(("Observation", "; ".join(errors)))

    correction_applied = bool(errors)
    if errors:
        for attempt in range(6):
            error_details = "\n".join(f"- {err}" for err in errors)
            react_steps.append(("Self-Correction", f"Correction {attempt + 1}/6 demandée : {error_details}"))
            correction_prompt = f"""Tu es un relecteur Python chargé d'une étape de Self-Correction.
Réécris entièrement les deux fichiers pour corriger les défauts signalés. Analyse silencieusement
le problème et ne révèle pas ta réflexion détaillée.

Contraintes obligatoires :
- code et tests doivent être des chaînes Python complètes et syntaxiquement valides ;
- tests doit commencer par `import pytest` et contenir au moins une fonction de module `def test_nom():` ;
- les fonctions importées dans tests doivent réellement exister dans code ;
- tests doit importer les fonctions depuis `solution.py` ;
- code doit commencer directement par un import, une définition ou une déclaration Python valide ;
- code et tests ne doivent contenir aucun texte explicatif, aucune balise Markdown et aucun préfixe `python` ;
- réécris les deux fichiers en version minimale, sans docstring ni chaîne multilignes ;
- chaque nom appelé dans un test doit être défini dans `solution.py` ; supprime tout test pour une fonction absente ;
- utilise des opérations Python simples et valides ; une addition utilise `x + y`, jamais `math.add(x, y)` ;
- ne crée pas de second nom comme `add_personnalise` si la demande ne le définit pas ;
- relis la sortie pytest et corrige précisément chaque `NameError`, `ImportError` et `AttributeError` ;
- conserve uniquement les tests nécessaires à la demande, sans test inventé supplémentaire ;
- tests doit rester court, avec des appels et des assertions simples ;
- n'utilise aucune chaîne multilignes, aucun retour à la ligne dans une chaîne et aucune f-string ;
- chaque chaîne doit avoir ses guillemets ouvrants et fermants sur la même ligne ;
- n'utilise aucun Markdown, aucune liste et aucun commentaire hors des chaînes de code ;
- analysis doit résumer les contraintes et les cas limites ;
- candidates doit contenir au moins deux approches distinctes ;
- selected_approach doit expliquer le choix effectué ;
- renvoie uniquement un JSON valide avec exactement les clés analysis, candidates, selected_approach, code, tests et explanation.

Défauts relevés :
{error_details}

Brouillon à rectifier :
{json.dumps(payload, ensure_ascii=False)}"""

            correction_response = client.chat.completions.create(
                model=model,
                max_tokens=max(max_tokens, 3600),
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Tu es un relecteur de code strict. Tu renvoies exclusivement un JSON corrigé."},
                    {"role": "user", "content": correction_prompt},
                ],
            )

            previous_payload = payload
            payload = normalize_payload(extract_json_payload(correction_response.choices[0].message.content))
            for field in ("analysis", "candidates", "selected_approach"):
                if field not in payload and field in previous_payload:
                    payload[field] = previous_payload[field]
            missing_fields = required_fields - payload.keys()
            if missing_fields:
                errors = [f"Champs JSON manquants : {', '.join(sorted(missing_fields))}"]
                continue

            errors = validate_reasoning_fields(payload) + validate_python(payload["code"], payload["tests"]) + validate_pytest_file(payload["tests"])
            if not errors:
                react_steps.append(("Analyse corrigée", payload["analysis"]))
                react_steps.append(("Tree of Thoughts corrigé", "Pistes envisagées :\n- " + "\n- ".join(payload["candidates"]) + "\n\nApproche retenue : " + payload["selected_approach"]))
                react_steps.append(("Action", "Relancer l'outil pytest après correction."))
                tests_passed, observation = run_pytest_tool(payload["code"], payload["tests"])
                react_steps.append(("Observation", observation))
                if not tests_passed:
                    errors = ["Échec de pytest :\n" + observation]
            if not errors:
                break

        if errors:
            raise ValueError("Échec de la validation après correction : " + "; ".join(errors))

    react_steps.append(("Réponse", "Code et tests validés, prêts à être affichés et téléchargés."))
    return GeneratedResult(
        code=payload["code"],
        tests=payload["tests"],
        explanation=payload["explanation"],
        analysis=payload["analysis"],
        candidates=payload["candidates"],
        selected_approach=payload["selected_approach"],
        react_steps=react_steps,
        correction_applied=correction_applied,
    )


def main() -> None:
    st.title("Programmateur")
    st.caption("Génération de code Python et de tests pytest avec un modèle local.")

    request = st.text_area(
        "Décrivez la fonctionnalité à développer",
        placeholder="Ex. Écrire une fonction qui vérifie si une chaîne est un palindrome et tester les cas limites.",
        height=140,
    )
    generate = st.button("Générer la solution", type="primary")

    if generate:
        if not request.strip():
            st.warning("Précisez une demande avant de lancer la génération.")
            return

        st.session_state.pop("result", None)
        with st.spinner("Génération, exécution des tests et correction en cours..."):
            try:
                result = ollama_generation(request)
                st.session_state["result"] = result
            except Exception as error:
                st.error(
                    "La génération Ollama a échoué. Aucun résultat de démonstration n'a été affiché. "
                    f"Détail : {type(error).__name__}: {error}"
                )
                return

    result = st.session_state.get("result")
    if result:
        st.info(result.explanation)

        with st.expander("Raisonnement de l'agent"):
            for label, detail in result.react_steps:
                st.markdown(f"**{label}** : {detail}")

        code_tab, tests_tab = st.tabs(["solution.py", "test_solution.py"])
        with code_tab:
            st.code(result.code, language="python")
            st.download_button("Télécharger solution.py", result.code, file_name="solution.py", mime="text/x-python")
        with tests_tab:
            st.code(result.tests, language="python")
            st.download_button("Télécharger test_solution.py", result.tests, file_name="test_solution.py", mime="text/x-python")


if __name__ == "__main__":
    main()