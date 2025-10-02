
### 1. The Universe of Obligation

*   **The Question it Answers:** **Who *should* have reported?**
*   **What it Contains:** This is our definitive list of every entity that we can identify from authoritative sources as having a legal obligation to submit a Modern Slavery Statement for a specific year.
*   **How We Build It:** We construct this universe by combining two key sources:
    *   **The ATO Corporate Tax Transparency data**, which gives us the list of high-revenue corporate entities.
    *   **The ACNC Charity Register data**, which gives us the list of high-revenue charities.
*   **Its Strategic Role:** This forms the **benchmark** against which we measure compliance.

### 2. The Universe of Action

*   **The Question it Answers:** **Who *did* report?**
*   **What it Contains:** This is our definitive list of every entity that has successfully *published* a Modern Slavery Statement on the official Register, regardless of whether they were legally obligated to or not.
*   **How We Build It:** We construct this universe by:
    1.  Taking the complete `All time data from Register.xlsx` file.
    2.  Filtering it to include **only** statements with a status of **'Published'**.
    3.  Using our **Universe of Identity** (see below) to clean the data and accurately link every published statement to a verified ABN.
*   **Its Strategic Role:** This is the factual record of all **compliant actions** that have been taken.

### 3. The Universe of Identity

*   **The Question it Answers:** **Who *is* this entity, definitively?**
*   **What it Contains:** This is our master "Rosetta Stone" or "phonebook". It is a comprehensive lookup table that links all known business identifiers together. It connects every registered business name to its official ABN and, where applicable, its ACN.
*   **How We Build It:** We construct this foundational universe by combining two key sources:
    *   **The ABN Bulk Extract**, which contains all registered ABNs and their associated names.
    *   **The ASIC Business Names Register**, which provides an additional layer of trading-name-to-entity links.
*   **Its Strategic Role:** This is the **enabling layer** that makes the entire strategy possible. It allows us to accurately connect the entities in the "Universe of Obligation" with the entities in the "Universe of Action," even when the names and identifiers in the raw data are messy and inconsistent.

---

### **How They Work Together**

Our final strategy works by taking the **Universe of Obligation** and the **Universe of Action**, and using the **Universe of Identity** to see where they overlap. The entities that exist in the "Universe of Obligation" but are missing from the "Universe of Action" are our definitive **non-compliers**.
