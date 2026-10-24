# Healthcare Data Visualization Plan

A plan for turning raw healthcare data into clear, accessible dashboards and reports that non-technical stakeholders (hospital executives, ward managers, quality leads and public-health officers) can use to make decisions.

**Author:** Ipshita Verma

## What this project contains

- A full written plan (Word and PDF) covering key healthcare metrics, tool selection and justification, the raw-data-to-visual process, a chart selection guide, user-centred design and accessibility, governance, and an implementation roadmap.
- Three dashboard mock-ups (hospital operations, clinical quality and safety, population health) and a one-page report wireframe.
- The Python script used to generate the mock-up images.

> **Note:** All data in the mock-ups is synthetic and for illustration only. It does not represent any real hospital or population.

## Preview

![Operations dashboard mock-up](images/fig2_exec.png)
![Quality dashboard mock-up](images/fig3_quality.png)

## Folder structure

| Folder | Contents |
|---|---|
| `docs/` | The full plan as `.docx` and `.pdf` |
| `images/` | Workflow diagram, dashboard mock-ups, report wireframe, colour palette |
| `scripts/` | `figs.py`: generates the images from synthetic data |

## How to run the script

```bash
pip install -r requirements.txt
cd scripts
python figs.py
```

The images are saved in the `images/` folder.

## Tools covered in the plan

Power BI (dashboards), Python with pandas and matplotlib (data preparation and statistical charts), PostgreSQL (data storage and modelling).

## Accessibility

The design uses a colour-blind-safe palette with shape and text cues in addition to colour, follows WCAG 2.1 AA guidance, and includes plain-language headlines on every chart.
