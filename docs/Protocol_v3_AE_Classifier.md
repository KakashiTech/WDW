# WDW v3.0: Protocolo Claro Autoencoder→Classifier

## Responde a Crítica 5: "Autoencoder vs classifier confusión"

---

## La Confusión Original

**Crítica del reviewer**: Si usas autoencoder para clasificación, necesitas explicar claramente cómo se hace clasificación y qué se optimiza.

**Problema**: En v1.0/v2.0, no estaba claro si:
- El autoencoder era el clasificador final
- La reconstrucción era el objetivo principal
- La clasificación era secundaria

---

## Protocolo Claro v3.0

### Arquitectura: Dos Componentes Distintos

```
┌─────────────────────────────────────────────────────────────┐
│                    WDW Classifier Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input (n-dim)                                              │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           WDW ENCODER (Feature Extractor)           │   │
│  │  ─────────────────────────────────────────────────  │   │
│  │  Quiver Propagation                                 │   │
│  │      ↓                                              │   │
│  │  Equivariant Projection (algebraic, non-learnable)   │   │
│  │      ↓                                              │   │
│  │  MERA Compression (learnable thetas)                │   │
│  │      ↓                                              │   │
│  │  Latent Representation (k-dim, k << n)              │   │
│  └─────────────────────────────────────────────────────┘   │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           CLASSIFIER HEAD (MLP Simple)              │   │
│  │  ─────────────────────────────────────────────────  │   │
│  │  FC: k → 128 → ReLU                                 │   │
│  │      ↓                                              │   │
│  │  FC: 128 → 10 (Softmax)                             │   │
│  └─────────────────────────────────────────────────────┘   │
│      ↓                                                      │
│  Output (10 clases)                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Función de Cada Componente

| Componente | Tipo | Función | Parámetros |
|------------|------|---------|------------|
| **WDW Encoder** | Feature Extractor | Transformar input a representación invariante rotacional | θ (MERA) + W_quiver |
| **Classifier Head** | MLP Estándar | Mapear features latentes a clases | W_fc1 + W_fc2 |

### ¿Qué Se Optimiza?

**Loss Total**: `L = L_classification + λ·L_equivariance + μ·L_reconstruction`

1. **`L_classification`** (Principal): Cross-entropy entre predicción y clase real
   - Esto es lo que realmente importa para el task
   - El encoder aprende features útiles para clasificación

2. **`L_equivariance`** (Regularización): Penalizar violaciones de equivariancia
   - Mantiene la propiedad de invarianza rotacional
   - No es el objetivo principal, es un constraint suave

3. **`L_reconstruction`** (Auxiliar): MSE entre input y reconstrucción
   - Ayuda a que el encoder preserve información
   - Peso μ es pequeño (0.01) para no dominar

### Pipeline de Entrenamiento

```julia
# 1. Forward Pass
latent = WDWEncoder(input)      # Features invariantes
logits = Classifier(latent)     # Predicción de clase
reconstructed = Decoder(latent) # Reconstrucción (opcional)

# 2. Calcular Loss
loss = cross_entropy(logits, target) +           # Principal
       λ * equivariance_penalty(latent) +        # Regularización  
       μ * mse(reconstructed, input)              # Auxiliar

# 3. Backward (gradient descent)
update!(encoder_params, ∇loss)
update!(classifier_params, ∇loss)
```

### Pipeline de Inferencia (Test)

```julia
# Solo se usa el encoder + classifier
# El decoder NO se usa en test para clasificación

latent = WDWEncoder(input)
class_probs = softmax(Classifier(latent))
predicted_class = argmax(class_probs)
```

---

## Analogía con Arquitecturas Estándar

| Arquitectura | Encoder | Classifier Head |
|-------------|---------|-----------------|
| **ResNet** | ResNet blocks (conv) | GlobalAvgPool + FC |
| **ViT** | Transformer patches | MLP head |
| **WDW (Ours)** | Quiver + Equivariant + MERA | MLP simple |

**WDW es un feature extractor** al igual que ResNet o ViT, solo que con:
- Propiedad: equivariancia algebraica (no aproximada)
- Eficiencia: O(n log n) parámetros vs O(n²)

---

## Métricas de Evaluación

### Primarias (Clasificación)
- **Accuracy**: Porcentaje de predicciones correctas
- **Precision/Recall**: Por clase
- **Confusion Matrix**: Error patterns

### Secundarias (Diagnóstico)
- **Equivariance Error**: ¿Mantiene invarianza?
- **Reconstruction Error**: ¿Preserva información?
- **Latent Complexity**: ¿Cuán comprimida es la representación?

---

## Por Qué Esto Responde la Crítica

### Antes (Confuso)
> "WDW es un autoencoder que también clasifica"

Problemas:
- ¿El objetivo es reconstrucción o clasificación?
- ¿Por qué usar autoencoder para clasificar?
- ¿Qué métrica importa?

### Ahora (Claro)
> "WDW es un feature extractor con propiedades algebraicas que alimenta un clasificador estándar"

Ventajas:
- Objetivo claro: clasificación con invarianza garantizada
- Encoder + Classifier es un patrón estándar (ResNet, ViT)
- Métrica principal: accuracy (como todo clasificador)

---

## Implementación en Código

```julia
struct WDWClassifier
    encoder::WDWEncoder      # Nuestro aporte
    classifier::MLP          # Estándar
end

function (model::WDWClassifier)(x)
    # Forward
    latent = model.encoder(x)
    return model.classifier(latent)
end

function loss(model, x, y_target)
    latent = model.encoder(x)
    logits = model.classifier(latent)
    
    # Principal: clasificación
    L_cls = crossentropy(logits, y_target)
    
    # Regularización: equivariancia
    L_equiv = equivariance_error(latent)
    
    return L_cls + λ * L_equiv
end
```

---

## Conclusión

**El WDW Autoencoder NO es el clasificador**—es el **feature extractor**.

**El clasificador es un MLP estándar** que opera sobre features invariantes.

**Esto es idéntico a ResNet/ViT**: arquitectura encoder + head, solo que nuestro encoder tiene propiedades algebraicas garantizadas.

---

## Checklist para el Paper

- [x] Arquitectura: Encoder + Classifier claramente separados
- [x] Objetivo: Clasificación (no reconstrucción)
- [x] Loss: Cross-entropy principal, equivariancia regularización
- [x] Métrica: Accuracy (comparación fair con baselines)
- [x] Analogía: ResNet/ViT pattern (lector familiar)
- [x] Diagrama: Pipeline claro (ver arriba)

**Estado: Crítica 5 completamente respondida.**
