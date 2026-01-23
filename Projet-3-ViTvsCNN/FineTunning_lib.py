import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random as rd
import time
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import copy
from thop import profile 

class BudgetAnalyzer:
    def __init__(self, device):
        self.device = device

    def get_flops_params(self, model, input_shape=(1, 3, 32, 32)):
        """ Calcule les FLOPs d'inférence et le nombre de paramètres """
        model_cpu = copy.deepcopy(model).cpu()
        model_cpu.eval()
        dummy_input = torch.randn(input_shape)
        try:
            # macs = Multiply-Accumulate. 1 MAC ≈ 2 FLOPs
            macs, params = profile(model_cpu, inputs=(dummy_input, ), verbose=False)
            flops_giga = (2 * macs) / 1e9
            params_million = params / 1e6
            return flops_giga, params_million
        except Exception as e:
            print(f"Erreur THOP: {e}")
            return 0, 0


    def measure_peak_memory(self, model, input_shape, mode='inference'):
        """ Mesure le pic de mémoire VRAM utilisé """
        if self.device.type != 'cuda': return 0

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model.to(self.device)
        dummy_input = torch.randn(input_shape).to(self.device)

        if mode == 'train':
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            
            
            output = model(dummy_input)
            
            # Si c'est un modèle Hugging Face, on prend les logits
            if hasattr(output, 'logits'):
                loss = output.logits.sum()
            else:
                loss = output.sum()
                
            loss.backward() 
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)

        peak_bytes = torch.cuda.max_memory_allocated(self.device)
        return peak_bytes / (1024**2) 

    def analyze_model(self, model, model_name, train_dataset_size, test_dataset_size, epochs, measured_train_time=None):
        """
        Génère le rapport complet.
        measured_train_time : Si vous avez mesuré le temps réel pendant votre boucle, passez-le ici.
                              Sinon, il sera estimé (moins précis).
        """
        results = {'Model': model_name}
        input_shape = (1, 3, 32, 32) 
        batch_shape = (64, 3, 32, 32) # Pour simuler la mémoire batch 64

        print(f"Analyse de {model_name}...")

        
        flops, params = self.get_flops_params(model, input_shape)
        results['Params (M)'] = params
        results['Inference FLOPs (G)'] = flops
        
        results['Train FLOPs (P)'] = (3 * flops * train_dataset_size * epochs) / 1e6

        results['Mem Test (MB)'] = self.measure_peak_memory(model, input_shape, mode='inference')
        # Pour le train, on mesure avec un batch de 64 
        results['Mem Train (MB)'] = self.measure_peak_memory(model, batch_shape, mode='train')

    
        sec_per_img = self.measure_time(model, input_shape, mode='inference')
        results['Time Test Total (s)'] = sec_per_img * test_dataset_size


        results['Time Train Total (min)'] = measured_train_time / 60
       

        return results
    


def compare_models(models_dict, input_shapes_dict, accuracies_dict, train_times_dict, test_times_dict, device):
    """
    Compare plusieurs modèles sur différentes métriques avec correction pour BatchNorm.
    """
    analyzer = BudgetAnalyzer(device)

    print("=" * 140)
    print("COMPARAISON DES MODÈLES")
    print("=" * 140)
    print()

    results = {}

    for model_name, model in models_dict.items():
        print(f"Analyse de {model_name}...")

        input_shape = input_shapes_dict.get(model_name)
        if input_shape is None:
            input_shape = (1, 3, 32, 32)
        
        print(f"  Input shape (inférence): {input_shape}")

    
        flops, params = analyzer.get_flops_params(model, input_shape)

        mem_inference = analyzer.measure_peak_memory(model, input_shape, mode='inference')

        # On force un batch de 32 pour l'adaptation BatchNorm
        train_input_shape = (32, input_shape[1], input_shape[2], input_shape[3])
        mem_train = analyzer.measure_peak_memory(model, train_input_shape, mode='train')

        test_acc = accuracies_dict.get(model_name, 0)
        if test_acc < 1:
            test_acc = test_acc * 100

        train_time = train_times_dict.get(model_name, 0)
        test_time = test_times_dict.get(model_name, 0)

        results[model_name] = {
            'input_shape': input_shape,
            'test_accuracy': test_acc,
            'flops_giga': flops,
            'params_million': params,
            'memory_inference_mb': mem_inference,
            'memory_train_mb': mem_train,
            'train_time_s': train_time,
            'test_time_s': test_time
        }

        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   FLOPs: {flops:.2f} GFLOPs")
        print(f"   Paramètres: {params:.2f} M")
        print(f"   Mémoire (inférence): {mem_inference:.2f} MB")
        print(f"   Mémoire (entraînement @batch 32): {mem_train:.2f} MB")
        print(f"   Temps d'entraînement: {train_time:.2f}s")
        print(f"   Temps de test: {test_time:.2f}s")
        print()

    # --- Tableau récapitulatif ---
    print("=" * 140)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 140)
    print(f"{'Modèle':<20} {'Input Shape':<18} {'Acc (%)':<12} {'FLOPs (G)':<12} {'Params (M)':<12} "
          f"{'Mem Inf (MB)':<15} {'Mem Train (MB)':<15} {'Train Time (s)':<15} {'Test Time (s)':<15}")
    print("-" * 140)

    for model_name, metrics in results.items():
        shape_str = str(metrics['input_shape'])
        print(f"{model_name:<20} {shape_str:<18} {metrics['test_accuracy']:<12.2f} {metrics['flops_giga']:<12.2f} "
              f"{metrics['params_million']:<12.2f} {metrics['memory_inference_mb']:<15.2f} "
              f"{metrics['memory_train_mb']:<15.2f} {metrics['train_time_s']:<15.2f} {metrics['test_time_s']:<15.2f}")

    print("=" * 140)

    return results