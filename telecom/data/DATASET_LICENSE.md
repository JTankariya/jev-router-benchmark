# Dataset Provenance & Licensing: Telecom Consumer Complaints Dataset

## 1. Overview
This dataset contains real-world subscriber support complaints and customer service inquiries filed against telecommunications providers, sourced from the **Comcast Consumer Complaints Dataset** on Kaggle.

The dataset documents subscriber disputes across broadband performance, service outages, billing errors, data caps, throttling, contract terms, and customer care interactions.

## 2. Official Licensing & Legal Status

- **Dataset Name:** Comcast Consumer Complaints Dataset
- **Source:** Kaggle (`archangan/comcast-complain-dataset`, `ccnm1987/comcast-consumer-complaints`)
- **Originating Data:** Public regulatory complaints and consumer dispute filings (FCC consumer reports)
- **Copyright Status:** **CC0 1.0 Universal (Public Domain Dedication)**
- **License Terms:**
  > *"To the extent possible under law, the author has dedicated all copyright and related and neighboring rights to this work to the public domain worldwide. This work is distributed without any warranty. You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission."*

## 3. Data Schema & Fields

| Field | Type | Description |
|---|---|---|
| `ticket_id` | String | Unique tracking identifier (e.g. `TEL-0001`) |
| `customer_complaint` | Text | Raw natural language subscriber complaint |
| `channel` | Categorical | Submission channel (`Customer Care Call`, `Internet`) |
| `city` / `state` | String | Subscriber geographic location |
| `target_department` | Categorical | Target customer support operational dispatch department |

## 4. Routing Taxonomy for Telecom Support Agents

The dataset benchmarks intent classification and routing across 4 core telecom operational support queues:

1. **`technical_network` (Technical Support & Network Operations):**
   * *Scope:* Slow broadband speeds, intermittent disconnects, full service outages, cable TV signal loss, modem/router hardware issues, packet loss.
2. **`billing_payments` (Billing & Financial Dispute Resolution):**
   * *Scope:* Overcharges, rate hikes, uncredited payments, unexpected fee additions, cancellation billing, refund requests, disconnection for disputed non-payment.
3. **`data_caps_policy` (Data Limits, Throttling & Usage Caps):**
   * *Scope:* 300GB/1TB monthly data caps, broadband speed throttling, overage penalty fees, fair use policy disputes, bandwidth limits on video streaming.
4. **`customer_retention` (Account Retention, Service Quality & Escalations):**
   * *Scope:* Cancellation requests, unfulfilled promotional promises, missed technician appointments, poor customer service agent interactions, deceptive sales tactics.

## 5. Privacy & Ethics
- All records in this dataset are public consumer dispute filings.
- No personally identifiable customer names, credit card numbers, phone numbers, or street addresses are included.
- Completely safe for open-source AI evaluation and benchmarking.
