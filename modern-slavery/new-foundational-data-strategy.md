
### **Our definitive strategy for mapping Modern Slavery Act compliance**

This document outlines our final, authoritative strategy for building a foundational dataset on Modern Slavery Act compliance. Our approach has been refined through a rigorous process of data discovery and validation.

The result will be a single, verified Master Compliance File. This file will provide the most complete and evidence-based view possible of the Australian modern slavery reporting landscape.

#### **Understanding the data landscape**

Our analysis confirmed that no single data source provides a complete picture of reporting obligations and actions. To address this, our strategy integrates three authoritative data sources to build a comprehensive view.

Each source serves a distinct and vital purpose.

*   **The Corporate Tax Transparency data** is the authority on which *corporate* entities were obligated to report in a given year. Its limitation is that it excludes other entity types, such as trusts and foreign entities.
*   **The Modern Slavery Register** is the authority on which entities *have* reported. This includes voluntary and non-corporate reporters. Its limitation is that it provides no context on an entity's legal obligation to report.
*   **The ABN Bulk Extract** is the authority on Australian business structures. It provides the crucial link between company names, ABNs and ACNs, allowing us to accurately connect the other two datasets.

Our strategy combines the strengths of these three sources to create a foundational dataset that is more powerful and conclusive than any single source.

#### **Our three-phase strategy**

We will execute the strategy in three phases.

**Phase 1: Build foundational data assets**

We will process our raw data into three distinct, clean and powerful assets.

| Task | Methodology | Final output |
| :--- | :--- | :--- |
| **Build the Universe of Identity** | We will process the complete ABN Bulk Extract to create a master lookup table. This will map every registered business name in Australia to its official ABN and ACN. | A clean `abn_master_lookup.csv` file. |
| **Build the Universe of Obligation** | We will consolidate the six annual Corporate Tax Transparency reports. This will create a definitive list of corporate entities that had a legal obligation to report for each year from 2019 to 2023. | A clean `obligated_entities.csv` file. |
| **Build the Universe of Action** | We will clean the Modern Slavery Register data. We will use our Universe of Identity asset to repair records and accurately identify the ABN for every statement submitted. | A clean and complete `reporting_history.csv` file. |

**Phase 2: Integrate the assets into a master file**

This is the core of our strategy. We will combine the three foundational assets into a single, entity-centric master file.

| Task | Methodology | Final output |
| :--- | :--- | :--- |
| **Create the entity superset** | We will create a master list of every unique ABN that appears in either the Universe of Obligation or the Universe of Action. | A skeleton dataframe of every relevant entity. |
| **Enrich the master file** | We will join the three foundational assets to this skeleton file. This will enrich each entity with its official identity, its year-by-year reporting obligations, and its complete reporting history. | A single, comprehensive master file. |

**Phase 3: Classify and conclude**

In the final phase, we will apply our business logic to derive a conclusive compliance status for every entity.

| Task | Methodology | Final output |
| :--- | :--- | :--- |
| **Classify compliance status** | We will apply a final set of rules to the master file. The logic will classify each entity as ‘Compliant’, ‘Non-Compliant’ or ‘Voluntary Reporter’ based on its unique obligation and reporting history. | **The Master Compliance File.** This is the definitive foundational dataset for all subsequent analysis. |

#### **The final output: a complete view**

This strategy will deliver a single, authoritative dataset that provides a panoramic view of the compliance landscape. It will allow us to:
*   confidently identify corporate entities that failed to meet their reporting obligations
*   correctly classify entities that reported voluntarily
*   understand the reporting patterns of non-corporate and foreign entities.

This foundational work will enable us to conduct targeted analysis and provide evidence-based insights to support the administration of the Modern Slavery Act.
