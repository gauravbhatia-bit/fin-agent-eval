---
doc_id: demobroker-fees
title: Fictional DemoBroker fee schedule
fictional_data: true
source_url: ""
retrieved_on: ""
verified: false
---

This document describes only a fictional fee schedule stored in `fees.yaml`. It is made-up data for repeatable software tests. It does not describe any real broker, account, market, or offer. The calculator applies the fixed and percentage components in that file, then applies the configured minimum and maximum. The returned estimate is denominated in the schedule's fictional currency.

For example, the invented ETF rule adds a fixed EUR 1.00 component and a percentage component of 0.10 percent of the order value, with a fictional minimum of EUR 1.00 and maximum of EUR 25.00. For a EUR 2,000 order, the percentage component is EUR 2.00, so the total before limits is EUR 3.00. These values exist only to make the formula testable; they must never be presented as real pricing.

The file does not model taxes, exchange fees, spreads, currency conversion costs, or other charges. A real provider's current official schedule would be needed for any actual cost. No transaction is submitted by this project.
