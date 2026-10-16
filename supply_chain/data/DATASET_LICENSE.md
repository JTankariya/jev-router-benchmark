# Dataset Provenance & Licensing: openFDA Food Enforcement Recall Dataset

## 1. Overview
This dataset contains real-world regulatory food recall and supply chain hazard notifications sourced directly from the **U.S. Food & Drug Administration (FDA) openFDA API** (`https://api.fda.gov/food/enforcement.json`).

These records document commercial food and beverage hazard events, carrier temperature excursions, pathogen surveillance findings, allergen cross-contaminations, and labeling non-compliance across nationwide supply chain networks.

## 2. Official Licensing & Legal Status

- **Source:** United States Food and Drug Administration (FDA), an operating division of the U.S. Department of Health and Human Services (HHS).
- **Endpoint:** `https://api.fda.gov/food/enforcement.json`
- **Copyright Status:** **Public Domain / CC0 Equivalent (U.S. Government Work)**
- **Governing Law:** Pursuant to **17 U.S.C. § 105**, works prepared by an officer or employee of the United States Government as part of that person's official duties are not eligible for copyright protection within the United States.
- **openFDA Terms of Service:**
  > *"openFDA is a public domain work under 17 U.S.C. § 105. It is free to use, share, and adapt without permission or licensing fees. There are no restrictions on commercial or non-commercial reproduction, redistribution, or derivation."*

## 3. Data Schema & Fields

| Field | Type | Description |
|---|---|---|
| `recall_number` | String | Official FDA recall tracking identifier (e.g. `F-0609-2015`) |
| `recalling_firm` | String | Commercial producer, processor, or supply chain distributor |
| `city` / `state` | String | Origin facility location |
| `product_description` | Text | Name, package size, lot number, UPC, and SKU |
| `reason_for_recall` | Text | Detailed explanation of hazard, pathogen, contamination, or defect |
| `distribution_pattern` | Text | States, nationwide channels, or retail networks receiving shipment |
| `fda_classification` | Categorical | Regulatory risk level (`Class I`, `Class II`, `Class III`) |
| `target_route` | Categorical | Mapped supply chain agent dispatch destination |
| `voluntary_mandated` | String | Recall initiation mechanism |

## 4. Routing Taxonomy for Supply Chain Agents

The dataset maps the 3 regulatory FDA risk tiers to standard supply chain operational action queues:

1. **`emergency_quarantine` (FDA Class I):**
   * *Hazard Profile:* Severe, life-threatening pathogens (*Listeria*, *Salmonella*, *E. Coli*, *Botulinum*), major undeclared allergens (peanuts, milk, eggs, tree nuts), lethal foreign objects (shards of glass or metal).
   * *Agent Action Queue:* Instantaneous freight carrier stop-shipment dispatch, warehouse dock lock, automated customer recall broadcast.
2. **`warehouse_hold` (FDA Class II):**
   * *Hazard Profile:* Reversible/moderate adverse health risks (*Burkholderia*, undeclared sulfites, sub-potency, cold-chain temperature excursion during transport, non-sharp foreign particles).
   * *Agent Action Queue:* Pallet quarantine at warehouse depot, secondary QA laboratory testing, supplier payment hold.
3. **`compliance_relabel` (FDA Class III):**
   * *Hazard Profile:* Technical labeling and packaging non-compliance unlikely to cause adverse health consequences (missing net weight, omitted language translation, unapproved herbicide trace within safety limits, expiration date format error).
   * *Agent Action Queue:* Administrative package relabeling, supplier compliance notification, no logistics stoppage.

## 5. Privacy & Ethics
- All company names, products, and recall summaries in this dataset are public regulatory filings released under the U.S. Freedom of Information Act and open data mandates.
- No personally identifiable information (PII) of consumers is present.
