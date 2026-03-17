"""
Modelo BERT con Transformers para procesamiento de lenguaje natural.
Incluye clasificación de textos, extracción de embeddings y fine-tuning.
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


class BERTClassifier:
    """
    Clasificador basado en BERT para tareas de clasificación de texto.
    Soporta modelos pre-entrenados de Hugging Face.
    """
    
    def __init__(self, model_name='bert-base-multilingual-cased', num_labels=2, device=None):
        """
        Inicializa el modelo BERT.
        
        Args:
            model_name (str): Nombre del modelo pre-entrenado de Hugging Face. 
                            Default: 'bert-base-multilingual-cased' (soporta múltiples idiomas)
                            Opciones: 'bert-base-uncased', 'bert-base-cased', 'distilbert-base-uncased', etc.
            num_labels (int): Número de clases para clasificación. Default: 2 (binaria)
            device (str): 'cuda' o 'cpu'. Si es None, detecta automáticamente.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Cargar tokenizador y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        
        print(f"📦 Modelo {model_name} cargado exitosamente.")
        print(f"🖥️  Dispositivo: {self.device}")
    
    def tokenize_texts(self, texts, max_length=128):
        """
        Tokeniza los textos para BERT.
        
        Args:
            texts (list): Lista de textos a tokenizar
            max_length (int): Longitud máxima del texto. Default: 128
            
        Returns:
            encodings (dict): Diccionario con 'input_ids', 'attention_mask' y 'token_type_ids'
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encodings
    
    def predict(self, texts, return_proba=False):
        """
        Realiza predicciones sobre textos.
        
        Args:
            texts (list): Lista de textos
            return_proba (bool): Si True, retorna probabilidades (softmax). Default: False
            
        Returns:
            predictions (list): Lista de predicciones (índices de clase)
            probabilities (list): Lista de probabilidades (si return_proba=True)
        """
        self.model.eval()
        
        encodings = self.tokenize_texts(texts)
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        if return_proba:
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            return predictions, probabilities
        
        return predictions
    
    def get_embeddings(self, texts, pooling='mean'):
        """
        Extrae embeddings (representaciones) de los textos.
        
        Args:
            texts (list): Lista de textos
            pooling (str): 'mean' para promedio de tokens, 'cls' para token [CLS]
            
        Returns:
            embeddings (np.ndarray): Array de embeddings con shape (num_textos, 768)
        """
        self.model.eval()
        
        encodings = self.tokenize_texts(texts)
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # Ultima capa
        
        if pooling == 'mean':
            # Promedio de tokens (ignora tokens de padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_hidden / sum_mask
        elif pooling == 'cls':
            # Token [CLS]
            embeddings = hidden_states[:, 0, :]
        else:
            raise ValueError("pooling debe ser 'mean' o 'cls'")
        
        return embeddings.cpu().numpy()
    
    def train_model(self, train_texts, train_labels, val_texts=None, val_labels=None,
                   epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Realiza fine-tuning del modelo BERT.
        
        Args:
            train_texts (list): Textos de entrenamiento
            train_labels (list): Etiquetas de entrenamiento
            val_texts (list): Textos de validación (opcional)
            val_labels (list): Etiquetas de validación (opcional)
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
            learning_rate (float): Tasa de aprendizaje
        """
        # Tokenizar
        train_encodings = self.tokenize_texts(train_texts)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
        
        # Crear dataset
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            train_labels_tensor
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Configurar optimizador
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Entrenamiento
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"📚 Época {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Validación
            if val_texts is not None and val_labels is not None:
                val_pred = self.predict(val_texts)
                accuracy = np.mean(val_pred == np.array(val_labels))
                print(f"✅ Validación - Accuracy: {accuracy:.4f}")
    
    def save_model(self, path):
        """Guarda el modelo entrenado."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"💾 Modelo guardado en: {path}")
    
    def load_model(self, path):
        """Carga un modelo previamente guardado."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"📂 Modelo cargado desde: {path}")


class BERTEmbeddings:
    """
    Extractor de embeddings usando BERT.
    Útil para similitud de textos, clustering, etc.
    """
    
    def __init__(self, model_name='bert-base-multilingual-cased', device=None):
        """Inicializa el modelo BERT para embeddings."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print(f"🔗 Modelo de embeddings {model_name} cargado.")
    
    def get_embedding(self, text):
        """Obtiene el embedding de un texto individual."""
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            embedding = last_hidden_state.mean(dim=1)  # Promedio de tokens
        
        return embedding.cpu().numpy()[0]
    
    def similarity(self, text1, text2):
        """
        Calcula similitud coseno entre dos textos.
        Rango: [-1, 1], donde 1 es idéntico.
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)


# ============= EJEMPLOS DE USO =============

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 EJEMPLOS DE USO DE BERT")
    print("=" * 60)
    
    # Ejemplo 1: Clasificación binaria (análisis de sentimiento)
    print("\n📊 EJEMPLO 1: Análisis de sentimiento")
    print("-" * 60)
    
    classifier = BERTClassifier(
        model_name='bert-base-multilingual-cased',
        num_labels=2
    )
    
    # Ejemplos de textos
    test_texts = [
        "Este producto es excelente, muy satisfecho",
        "Muy malo, no lo recomiendo"
    ]
    
    predictions = classifier.predict(test_texts, return_proba=True)
    pred_labels, probabilities = predictions
    
    for text, pred, proba in zip(test_texts, pred_labels, probabilities):
        sentiment = "Positivo" if pred == 1 else "Negativo"
        print(f"📝 '{text}'")
        print(f"   ➜ {sentiment} (confianza: {max(proba):.2%})\n")
    
    # Ejemplo 2: Extracción de embeddings
    print("\n🔗 EJEMPLO 2: Extracción de embeddings")
    print("-" * 60)
    
    embeddings = classifier.get_embeddings(test_texts)
    print(f"✅ Embeddings extraídos: shape {embeddings.shape}")
    print(f"   Dimensión de cada embedding: {embeddings[0].shape}")
    
    # Ejemplo 3: Similitud entre textos
    print("\n📏 EJEMPLO 3: Similitud entre textos")
    print("-" * 60)
    
    bert_emb = BERTEmbeddings(model_name='bert-base-multilingual-cased')
    
    text1 = "El gato está durmiendo"
    text2 = "El gato está dormido"
    text3 = "El perro corre rápido"
    
    sim_12 = bert_emb.similarity(text1, text2)
    sim_13 = bert_emb.similarity(text1, text3)
    
    print(f"'{text1}'")
    print(f"  ↔ '{text2}'")
    print(f"    Similitud: {sim_12:.4f} ✅ (muy similar)\n")
    
    print(f"'{text1}'")
    print(f"  ↔ '{text3}'")
    print(f"    Similitud: {sim_13:.4f} ❌ (diferente)\n")
    
    print("=" * 60)
    print("✨ ¡BERT configurado correctamente!")
    print("=" * 60)
