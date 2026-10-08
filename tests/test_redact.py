from jobagent.resume.redact import redact_pii


def test_redacts_contact_details():
    out = redact_pii("Jane jane.doe@x.com +91 98765 43210 https://github.com/jane Python")
    assert "jane.doe@x.com" not in out and "98765" not in out and "github.com/jane" not in out
    assert "Python" in out
