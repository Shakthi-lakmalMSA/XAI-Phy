import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class InsightEngine:
    """
    The core engine for running physics-based simulations on sentence tokens
    to generate interpretability maps for Large Language Models.
    """
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initializes the tokenizer and model.

        Args:
            model_name (str): The name of the transformer model to use from Hugging Face.
        """
        print(f"Initializing engine with model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        print("Engine initialized successfully.")

    def _get_embeddings_and_attention(self, text):
        """
        Extracts token embeddings and the attention matrix for a given text.

        Args:
            text (str): The input text to analyze.

        Returns:
            tuple: A tuple containing:
                - tokens (list): A list of tokens.
                - token_embeddings (np.array): A numpy array of token embeddings.
                - attention_matrix (np.array): The model's attention matrix, averaged across all heads.
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        token_embeddings = outputs.last_hidden_state[0].numpy()
        
        # Average attention heads for simplicity
        attention = outputs.attentions[-1]  # Last layer attentions
        attention_matrix = torch.mean(attention[0], axis=0).numpy()

        return tokens, token_embeddings, attention_matrix

    def _run_simulation(self, token_embeddings, attention_matrix, sim_params):
        """
        Runs the core physics-based simulation.

        Args:
            token_embeddings (np.array): Embeddings for each token.
            attention_matrix (np.array): Attention matrix from the model.
            sim_params (dict): A dictionary of simulation parameters.

        Returns:
            np.array: The final positions of the tokens after the simulation.
        """
        num_particles = len(token_embeddings)
        positions = np.random.rand(num_particles, 2) * 100
        velocities = np.zeros((num_particles, 2))
        
        # Calculate semantic similarity matrix once
        semantic_sim = cosine_similarity(token_embeddings)

        for i in range(sim_params['iterations']):
            net_forces = np.zeros((num_particles, 2))
            for p1 in range(num_particles):
                for p2 in range(num_particles):
                    if p1 == p2:
                        continue

                    # Calculate vector and distance
                    direction_vec = positions[p2] - positions[p1]
                    dist = np.linalg.norm(direction_vec)
                    if dist < 1e-6: continue
                    unit_vec = direction_vec / dist
                    
                    # 1. Semantic Force (Attraction/Repulsion)
                    sim = semantic_sim[p1, p2]
                    semantic_force_magnitude = 0
                    if sim > sim_params['semantic_attraction_threshold']:
                        semantic_force_magnitude = sim_params['semantic_force_strength'] * (sim - sim_params['semantic_attraction_threshold'])
                    elif sim < sim_params['semantic_repulsion_threshold']:
                        semantic_force_magnitude = -sim_params['semantic_force_strength'] * (sim_params['semantic_repulsion_threshold'] - sim)
                    
                    # 2. Contextual Attention Force (Attraction only)
                    attention_force_magnitude = sim_params['attention_force_strength'] * attention_matrix[p1, p2]
                    
                    # Combine forces
                    total_force = (semantic_force_magnitude + attention_force_magnitude) * unit_vec
                    net_forces[p1] += total_force
            
            # Update kinematics
            velocities = (velocities + net_forces) * sim_params['drag']
            positions += velocities

        return positions

    def analyze_sentence(self, text, **kwargs):
        """
        Analyzes a single sentence to generate a reasoning map.

        Args:
            text (str): The sentence to analyze.
            **kwargs: Custom simulation parameters can be passed here.

        Returns:
            dict: A dictionary containing the analysis results.
        """
        sim_params = {
            'iterations': 200,
            'semantic_force_strength': 0.5,
            'attention_force_strength': 2.0,
            'semantic_attraction_threshold': 0.6,
            'semantic_repulsion_threshold': 0.3,
            'drag': 0.95
        }
        sim_params.update(kwargs) # Allow overriding default params

        tokens, embeddings, attention = self._get_embeddings_and_attention(text)
        
        final_positions = self._run_simulation(embeddings, attention, sim_params)

        return {
            'text': text,
            'tokens': tokens,
            'token_embeddings': embeddings,
            'attention_matrix': attention,
            'final_positions': final_positions,
            'simulation_parameters': sim_params
        }
