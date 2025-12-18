from pathlib import Path
import urllib.request

SAMPLES = [
    ("nda_harvard.pdf", "https://otd.harvard.edu/uploads/Files/Template_CDA_-_Mutual_Disclosure_-_One_PI_-_Oct_2018.pdf"),
    ("nda_ut_austin.pdf", "https://sites.utexas.edu/moriba/files/2017/04/UT_Universal_NDA_2014_OIE.pdf"),
    ("nda_torys.pdf", "https://www.torys.com/-/media/project/zenith-tenant/zenith-site/assets/startup-legal-playbook/sample-docs/unilateral-nda-template-ecvc.pdf"),
    ("saas_mantal.pdf", "https://app.mantal.co.uk/policies/docs/Mantal-M-SAAS-001-SaaS.pdf"),
    ("saas_fieldwire.pdf", "https://www.fieldwire.com/legal/en/subscription_agreement.pdf"),
    ("msa_mercycorps.pdf", "https://www.mercycorps.org/sites/default/files/2020-01/Attachment%202%20-%20Sample%20Master%20Service%20Agreement.pdf"),
]

out_dir = Path("data/raw_pdfs")
out_dir.mkdir(parents=True, exist_ok=True)

for filename, url in SAMPLES:
    dest = out_dir / filename
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

print("Done.")
