"""Small Streamlit frontend for the FastAPI agent backend."""
from __future__ import annotations

import os
from datetime import date

import httpx
import streamlit as st


st.set_page_config(page_title="Everest Student Services", page_icon="🎓")
st.title("Everest Student Services")
st.caption("Ask a university question. The backend routes it and returns grounded results.")

api_url = st.sidebar.text_input(
    "Backend API URL",
    value=os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
)
st.sidebar.markdown(f"API docs: [{api_url}/docs]({api_url}/docs)")

if "student_id" not in st.session_state:
    st.session_state.student_id = ""
if "needs_student_id" not in st.session_state:
    st.session_state.needs_student_id = False
if "pending_request" not in st.session_state:
    st.session_state.pending_request = None
if "answer_payload" not in st.session_state:
    st.session_state.answer_payload = None
if "ui_error" not in st.session_state:
    st.session_state.ui_error = ""
if "route_intent" not in st.session_state:
    st.session_state.route_intent = ""


def post_json(path: str, payload: dict, headers: dict[str, str] | None = None):
    try:
        response = httpx.post(
            f"{api_url}{path}",
            json=payload,
            headers=headers,
            timeout=300,
        )
    except httpx.HTTPError as error:
        return None, f"Could not reach the backend at {api_url}: {error}"

    try:
        data = response.json()
    except ValueError:
        data = {"detail": response.text}
    if not response.is_success:
        detail = data.get("detail", data)
        return None, f"Backend returned {response.status_code}: {detail}"
    return data, ""


def ask_backend(payload: dict, student_id: str = "") -> None:
    headers = {"X-Student-Id": student_id} if student_id else None
    answer, error = post_json("/ask", payload, headers)
    st.session_state.answer_payload = answer
    st.session_state.ui_error = error
    if not error:
        st.session_state.needs_student_id = False
        st.session_state.pending_request = None


with st.form("ask_form"):
    question = st.text_area(
        "Your question",
        placeholder="What branches are offered in the B.Tech programme?",
        height=100,
    )
    as_of_date = st.date_input("Apply rules as of", value=date.today())
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    st.session_state.answer_payload = None
    st.session_state.route_intent = ""
    if not question.strip():
        st.session_state.ui_error = "Enter a question first."
    else:
        request_payload = {"question": question.strip()}
        if as_of_date:
            request_payload["as_of_date"] = as_of_date.isoformat()

        route_data, route_error = post_json("/route", request_payload)
        st.session_state.ui_error = route_error
        if route_data:
            st.session_state.route_intent = route_data["intent"]
            if route_data["requires_student_id"] and not st.session_state.student_id.strip():
                st.session_state.pending_request = request_payload
                st.session_state.needs_student_id = True
                st.session_state.ui_error = "Student ID is required for this personal-data query."
            else:
                identity = (
                    st.session_state.student_id.strip()
                    if route_data["requires_student_id"]
                    else ""
                )
                ask_backend(request_payload, identity)

if st.session_state.route_intent:
    st.caption(f"Backend router detected: `{st.session_state.route_intent}`")

if st.session_state.needs_student_id:
    st.info("The backend identified a student-data query. Enter the authenticated student ID to continue.")
    st.text_input("Student ID", placeholder="S1008", key="student_id")
    if st.button("Continue", type="primary"):
        if not st.session_state.student_id.strip():
            st.session_state.ui_error = "Student ID is required for this personal-data query."
        elif st.session_state.pending_request:
            ask_backend(
                st.session_state.pending_request,
                st.session_state.student_id.strip(),
            )

if st.session_state.ui_error:
    st.error(st.session_state.ui_error)

payload = st.session_state.answer_payload
if payload:
    st.subheader(payload.get("answer_type", "Answer").replace("_", " ").title())
    st.write(payload.get("answer", ""))
    st.caption(f"Trace ID: {payload.get('trace_id', '')} · As of: {payload.get('as_of_date', '')}")

    citations = payload.get("citations", [])
    tools = payload.get("tools_invoked", [])
    rules = payload.get("applied_rules", [])
    conflicts = payload.get("conflicts_detected", [])
    if citations or tools or rules or conflicts:
        with st.expander("Sources and tool details"):
            st.json({
                "citations": citations,
                "tools_invoked": tools,
                "applied_rules": rules,
                "conflicts_detected": conflicts,
            })
