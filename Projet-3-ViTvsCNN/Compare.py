import torch
import time
import copy
import pandas as pd
from thop import profile

class BudgetAnalyzer:
    def __init__(self, device):
        self.device = device
#Fonction pour calculer les FLOPs nombre d'opérations mathématiques et de paramètres
    
    def get_flops_params(self, model, input_shape=(1, 3, 32, 32)):
      
        model_cpu = copy.deepcopy(model).cpu() 
        model_cpu.eval()  #pour pas modifier le modèle
        dummy_input = torch.randn(input_shape) # Entrée fictive avec des valeurs aléatoires de la taille de l'image fournie (input_shape)
        
        try:
            #  1 MAC ≈ 2 FLOPs
            macs, params = profile(model_cpu, inputs=(dummy_input, ), verbose=False) #profile compte le nombre d'opérations et paramètres
            flops_giga = (2 * macs) / 1e9
            params_million = params / 1e6
            return flops_giga, params_million
        except Exception as e:
            print(f"Erreur THOP: {e}")
            return 0, 0

# Fonction mesurant le pic de mémoire pendant un entrainement ou test
    def measure_peak_memory(self, model, input_shape=(1, 3, 32, 32), mode='inference'):
        """ Mesure le pic de mémoire VRAM utilisé """
        if self.device.type != 'cuda': return 0

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model.to(self.device)
        dummy_input = torch.randn(input_shape).to(self.device) # Input aléatoire de même dimension que l'image

        if mode == 'train':
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()
            loss = model(dummy_input).sum()
            loss.backward() 
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)

        peak_bytes = torch.cuda.max_memory_allocated(self.device)# Récupère la mémoire max allouée
        return peak_bytes / (1024**2) # En MB
        
#Génération d'un rapport complet
    def analyze_model(self, model, model_name, train_dataset_size, test_dataset_size, epochs, measured_train_time=None):

        results = {'Model': model_name}
        input_shape = (1, 3, 32, 32) # Une seule image
        batch_shape = (64, 3, 32, 32) # Pour simuler la mémoire batch

        print(f"Analyse de {model_name}...")

        # 1. FLOPs & Params
        flops, params = self.get_flops_params(model, input_shape)
        results['Params (M)'] = params
        results['Inference FLOPs (G)'] = flops
        # Estimation FLOPs Train: 3 * Inference * Nb_Images * Nb_Epoques
        results['Train FLOPs (P)'] = (3 * flops * train_dataset_size * epochs) / 1e6

        # 2. Mémoire (VRAM)
        results['Mem Test (MB)'] = self.measure_peak_memory(model, input_shape, mode='inference')
        # Pour le train, on mesure avec un batch de 64 
        results['Mem Train (MB)'] = self.measure_peak_memory(model, batch_shape, mode='train')

        # 3. Temps (Time)
        # Temps Test (Total pour tout le dataset de test)
        sec_per_img = self.measure_time(model, input_shape, mode='inference')
        results['Time Test Total (s)'] = sec_per_img * test_dataset_size

        # Temps Train (Total)
        if measured_train_time is not None:
            results['Time Train Total (min)'] = measured_train_time / 60
        else:
            # Estimation si non fourni (moins précis car ignore le chargement des données)
            sec_per_batch = self.measure_time(model, batch_shape, mode='train')
            batches_per_epoch = train_dataset_size / 64
            total_sec = sec_per_batch * batches_per_epoch * epochs
            results['Time Train Total (min)'] = total_sec / 60

        return results


def compare_models(models_dict, input_shapes_dict, accuracies_dict, train_times_dict, device):

    analyzer = BudgetAnalyzer(device)

    print("=" * 140)
    print("COMPARAISON DES MODÈLES")
    print("=" * 140)
    print()

    results = {}

    for model_name, model in models_dict.items():
        print(f"Analyse de {model_name}...")

        # Récupère l'input shape spécifique au modèle
        input_shape = input_shapes_dict.get(model_name)
        if input_shape is None:
            print(f"  Aucune input_shape fournie pour {model_name}, utilisation par défaut (1, 3, 32, 32)")
            input_shape = (1, 3, 32, 32)
        else:
            print(f"  Input shape: {input_shape}")

        # Calcul des FLOPs et paramètres
        flops, params = analyzer.get_flops_params(model, input_shape)

        # Mémoire en inférence
        mem_inference = analyzer.measure_peak_memory(model, input_shape, mode='inference')

        # Mémoire en entraînement
        mem_train = analyzer.measure_peak_memory(model, input_shape, mode='train')

        # Récupérer l'accuracy
        test_acc = accuracies_dict.get(model_name, 0)
        if test_acc < 1:
            test_acc = test_acc * 100

        # Récupérer les temps
        train_time = train_times_dict.get(model_name, 0)


        results[model_name] = {
            'input_shape': input_shape,
            'test_accuracy': test_acc,
            'flops_giga': flops,
            'params_million': params,
            'memory_inference_mb': mem_inference,
            'memory_train_mb': mem_train,
            'train_time_s': train_time,
        }

        print(f"  ✓ Test Accuracy: {test_acc:.2f}%")
        print(f"  ✓ FLOPs: {flops:.2f} GFLOPs")
        print(f"  ✓ Paramètres: {params:.2f} M")
        print(f"  ✓ Mémoire (inférence): {mem_inference:.2f} MB")
        print(f"  ✓ Mémoire (entraînement): {mem_train:.2f} MB")
        print(f"  ✓ Temps d'entraînement: {train_time:.2f}s")
        print()

    # Tableau récapitulatif
    print("=" * 140)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 140)
    print(f"{'Modèle':<20} {'Input Shape':<18} {'Acc (%)':<12} {'FLOPs (G)':<12} {'Params (M)':<12} "
          f"{'Mem Inf (MB)':<15} {'Mem Train (MB)':<15} {'Train Time (s)':<15}")
    print("-" * 140)

    for model_name, metrics in results.items():
        shape_str = str(metrics['input_shape'])
        print(f"{model_name:<20} {shape_str:<18} {metrics['test_accuracy']:<12.2f} {metrics['flops_giga']:<12.2f} "
              f"{metrics['params_million']:<12.2f} {metrics['memory_inference_mb']:<15.2f} "
              f"{metrics['memory_train_mb']:<15.2f} {metrics['train_time_s']:<15.2f}")

    print("=" * 140)

    return results


