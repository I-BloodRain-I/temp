import torch
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import contextmanager
from collections import defaultdict
import pandas as pd

class CUDAProfiler:
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        self.stats = defaultdict(lambda: {
            'cpu_time': 0.0,
            'cuda_time': 0.0,
            'cpu_kernel_percent': 0.0,
            'memory_allocated': 0.0,
            'memory_reserved': 0.0,
            'cuda_memory_accessed': 0.0,
            'cuda_memory_allocated': 0.0,
            'calls': 0
        })
        
    @contextmanager
    def profile_section(self, section_name):
        """Context manager for profiling a specific section of code"""
        if not self.enabled:
            yield
            return

        # Включаем сбор всех типов активности
        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ]
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function(section_name):
                yield
        
        # Собираем статистику
        events = prof.key_averages()
        total_time = sum(evt.cpu_time_total for evt in events)
        cuda_total_time = sum(evt.cuda_time_total for evt in events)
        
        # Анализируем использование памяти
        cuda_mem_allocated = torch.cuda.memory_allocated()
        cuda_mem_reserved = torch.cuda.memory_reserved()
        
        # Собираем статистику по операциям
        cuda_kernel_time = sum(evt.cuda_time_total for evt in events if "cuda" in evt.key.lower())
        cpu_kernel_time = sum(evt.cpu_time_total for evt in events if "cpu" in evt.key.lower())
        
        # Вычисляем проценты
        total_compute_time = cpu_kernel_time + cuda_kernel_time
        cpu_percent = (cpu_kernel_time / total_compute_time * 100) if total_compute_time > 0 else 0
        
        # Обновляем статистику
        self.stats[section_name]['cpu_time'] += cpu_kernel_time / 1000  # convert to ms
        self.stats[section_name]['cuda_time'] += cuda_kernel_time / 1000
        self.stats[section_name]['cpu_kernel_percent'] = cpu_percent
        self.stats[section_name]['memory_allocated'] = cuda_mem_allocated / (1024 * 1024)  # MB
        self.stats[section_name]['memory_reserved'] = cuda_mem_reserved / (1024 * 1024)  # MB
        self.stats[section_name]['calls'] += 1
        
        # Выводим детальную информацию о событиях
        print(f"\nProfiling results for {section_name}:")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        ))

    def get_stats_df(self):
        """Get profiling statistics as a pandas DataFrame"""
        data = []
        for section, metrics in self.stats.items():
            data.append({
                'Section': section,
                'Calls': metrics['calls'],
                'CPU Time (ms)': metrics['cpu_time'],
                'CUDA Time (ms)': metrics['cuda_time'],
                'CPU Kernel %': metrics['cpu_kernel_percent'],
                'GPU Memory Allocated (MB)': metrics['memory_allocated'],
                'GPU Memory Reserved (MB)': metrics['memory_reserved'],
                'Avg Time/Call (ms)': (metrics['cpu_time'] + metrics['cuda_time']) / metrics['calls']
            })
        
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include=['float64']).columns
        df[numeric_cols] = df[numeric_cols].round(2)
        
        return df.sort_values('CUDA Time (ms)', ascending=False)

    def print_memory_stats(self):
        """Print current and peak memory usage"""
        current_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        current_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        
        print(f"\nCurrent GPU Memory Allocated: {current_allocated:.2f} MB")
        print(f"Current GPU Memory Reserved: {current_reserved:.2f} MB")
        print(f"Peak GPU Memory Allocated: {max_allocated:.2f} MB")
        print(f"Peak GPU Memory Reserved: {max_reserved:.2f} MB")

    def reset(self):
        """Reset all profiling statistics"""
        self.stats.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()