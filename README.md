# Medical Knowledge Graph - Proof of Concept (POC)

## Overview
This repository contains a Proof of Concept (POC) implementation of a medical knowledge graph generation system. The goal of this project is to demonstrate the feasibility of analyzing medical documents and generating a knowledge graph based on the extracted biomedical entities and their relationships.

## Features
- Extracts named entities from medical documents using SpaCy with a domain-specific medical model.
- Identifies biomedical entities using a pre-trained NER model.
- Generates simple relationships between entities based on proximity.
- Visualizes the knowledge graph using NetworkX and Matplotlib.
- Provides entity visualization within the medical document using SpaCy's entity visualizer.

## Requirements
- Python 3.12
- SpaCy
- NLTK
- NetworkX
- Matplotlib
