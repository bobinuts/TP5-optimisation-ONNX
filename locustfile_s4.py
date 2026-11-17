"""locustfile_s4.py - Seance 4 : charge sur l'API d'inference S4 (CNN sur vignettes).

Reprend le scenario de la seance 3 (locustfile.py) : memes temps de reflexion, memes routes,
meme poids. Seule la tache de prediction change : elle envoie l'image de controle en multipart.

Usage (palier 1) :
    IMAGE_CONTROLE_S4=controle.jpg locust -f locustfile_s4.py --headless -u 10 -r 2 -t 3m \
        --host "$APP_URL" --csv palier_s4_1 --html palier_s4_1.html

Controle d'un appel unique AVANT le premier palier (doit afficher 10 x 200) :
    IMAGE_CONTROLE_S4=controle.jpg python locustfile_s4.py --check "$APP_URL"
"""
import os
import sys
import uuid

from locust import HttpUser, between, task

# L'image est lue UNE fois, au chargement du module : on mesure le service, pas le disque du poste.
CHEMIN_IMAGE = os.environ.get("IMAGE_CONTROLE_S4")
if not CHEMIN_IMAGE or not os.path.isfile(CHEMIN_IMAGE):
    raise SystemExit("IMAGE_CONTROLE_S4 doit designer le fichier de l'image de controle (.jpg)")
with open(CHEMIN_IMAGE, "rb") as f:
    IMAGE = f.read()


class Dermatologue(HttpUser):
    """Un praticien qui soumet des lesions depuis son poste de consultation."""

    # Celui de S3 : 1 a 5 s de reflexion, pour comparer a usage egal.
    wait_time = between(1, 5)

    # Poids 10 / 1 / 1, comme en S3 : /predict domine, les deux routes de supervision sont rares.
    @task(10)
    def predire(self):
        with self.client.post(
            "/predict",
            files={"image": ("controle.jpg", IMAGE, "image/jpeg")},
            headers={"X-Request-ID": str(uuid.uuid4())},
            name="POST /predict",
            catch_response=True,
        ) as reponse:
            if reponse.status_code == 200:
                reponse.success()
            else:
                reponse.failure(f"statut {reponse.status_code}")

    @task(1)
    def sante(self):
        self.client.get("/health", name="GET /health")

    @task(1)
    def info_modele(self):
        self.client.get("/model/info", name="GET /model/info")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--check":
        raise SystemExit("usage : python locustfile_s4.py --check <APP_URL>")
    import requests

    url = sys.argv[2].rstrip("/")
    codes = []
    for _ in range(10):
        r = requests.post(f"{url}/predict", files={"image": ("controle.jpg", IMAGE, "image/jpeg")}, timeout=60)
        codes.append(r.status_code)
    print("codes /predict :", codes)
    sys.exit(0 if all(c == 200 for c in codes) else 1)
