# Multi-Agent Work Breakdown

## Agent A — Data + Preprocessing
Phases 1–2.  
Input: dataset.  
Output: loaded/scaled data + scaler params + CP1/CP2.

## Agent B — PCA Core
Phases 3–4.  
Input: Agent A outputs.  
Output: PCAFromScratch, eigenbasis, validation metrics + CP3.

## Agent C — Evaluation
Phases 5–8.  
Input: Agent B outputs.  
Output: k thresholds, plots, reconstruction, loadings + CP4–CP7.

## Agent D — Selection + Export
Phases 9–11.  
Input: Agent C outputs.  
Output: PCA90/PCA95 + exported artifacts + CP8–CP10.

## Agent E — Classification
Input: Original standardized, PCA90, PCA95.  
Không được fit PCA lại hoặc đổi split.
