# Architecture

The reference flow deliberately separates untrusted transformation steps from policy enforcement:

```text
ingestion -> de-identification -> extraction -> grounding -> review -> FHIR proposal
             (interface only —                              |
              not implemented)                         persistence guard
```

De-identification is specified but not implemented in this repository; see
`pipelines/deidentification/README.md` for the required two-phase interface and its
validation burden. The synthetic demo needs no de-identification, and no stage in this
repository performs one.

Provider responses are candidate data only. Grounding verifies exact evidence before a candidate reaches review. The reviewer approves or rejects a candidate; only an approved candidate can become a FHIR proposal. Persistence remains disabled unless the deployment separately enables it and provides an audited FHIR adapter.

![Clinical note to grounded facts to FHIR pipeline](images/clinical-note-grounded-facts-fhir-pipeline.png)
