"""L'interface se lance sans erreur et le formulaire produit bien une alerte."""
from streamlit.testing.v1 import AppTest


def test_application_se_lance():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
    assert [m.value for m in at.metric][1] == "21"  # alertes en 1996


def test_formulaire_contrat_recent_donne_une_alerte():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    at.selectbox[2].set_value("1 to 7")  # accident 1 à 7 jours après la souscription
    at.button[0].click().run()
    assert not at.exception
    assert any('badge b-alerte' in m.body for m in at.markdown)
