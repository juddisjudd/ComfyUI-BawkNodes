"""
Smart Cache Manager for ComfyUI Diffusion Model Loader
File: modules/cache_manager.py
"""

import time
import threading
import weakref
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
import comfy.model_management as mm


class SmartCacheManager:
    """
    Advanced caching system with:
    - Memory-aware LRU eviction
    - Automatic cleanup
    - Thread-safe operations
    - Memory pressure detection
    """
    
    def __init__(self, max_memory_gb: float = 8.0, max_items: int = 10):
        """
        Initialize cache manager
        
        Args:
            max_memory_gb: Maximum memory to use for caching (GB)
            max_items: Maximum number of items to cache
        """
        self.max_memory = max_memory_gb * (1024**3)  # Convert to bytes
        self.max_items = max_items
        
        # Thread-safe cache storage
        self._cache = OrderedDict()
        self._cache_sizes = {}
        self._access_times = {}
        self._current_memory = 0
        self._lock = threading.RLock()
        
        # Weak references for automatic cleanup
        self._weak_refs = {}
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get_cached(self, key: str) -> Optional[Any]:
        """
        Get item from cache if it exists
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Update access time and move to end (most recent)
                self._access_times[key] = time.time()
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            
            self._misses += 1
            return None
    
    def cache_result(self, key: str, result: Any, estimated_size: int = 0) -> bool:
        """
        Cache a result with memory management
        
        Args:
            key: Cache key
            result: Result to cache
            estimated_size: Estimated memory size in bytes
            
        Returns:
            True if cached successfully, False otherwise
        """
        with self._lock:
            # Check if we need to evict items
            if not self._make_space(estimated_size):
                print(f"[CacheManager] Cannot cache item {key}: insufficient space")
                return False
            
            # Remove existing item if updating
            if key in self._cache:
                self._remove_item(key)
            
            # Add new item
            self._cache[key] = result
            self._cache_sizes[key] = estimated_size
            self._access_times[key] = time.time()
            self._current_memory += estimated_size
            
            # Set up weak reference for automatic cleanup
            try:
                self._weak_refs[key] = weakref.ref(result, lambda ref: self._cleanup_weak_ref(key))
            except TypeError:
                # Some objects can't have weak references
                pass
            
            return True
    
    def _make_space(self, needed_space: int) -> bool:
        """
        Make space in cache by evicting items if necessary
        
        Args:
            needed_space: Space needed in bytes
            
        Returns:
            True if space was made available, False otherwise
        """
        # Check if we have enough space
        if (self._current_memory + needed_space <= self.max_memory and 
            len(self._cache) < self.max_items):
            return True
        
        # Sort items by access time (oldest first)
        items_by_age = sorted(
            self._access_times.items(),
            key=lambda x: x[1]
        )
        
        # Evict oldest items until we have enough space
        for key, _ in items_by_age:
            if key in self._cache:
                self._remove_item(key)
                self._evictions += 1
                
                # Check if we now have enough space
                if (self._current_memory + needed_space <= self.max_memory and 
                    len(self._cache) < self.max_items):
                    return True
        
        # If we still don't have space after evicting everything, the item is too large
        return self._current_memory + needed_space <= self.max_memory
    
    def _remove_item(self, key: str):
        """Remove an item from all cache structures"""
        if key in self._cache:
            del self._cache[key]
        
        if key in self._cache_sizes:
            self._current_memory -= self._cache_sizes[key]
            del self._cache_sizes[key]
        
        if key in self._access_times:
            del self._access_times[key]
        
        if key in self._weak_refs:
            del self._weak_refs[key]
    
    def _cleanup_weak_ref(self, key: str):
        """Cleanup when a weak reference is garbage collected"""
        with self._lock:
            if key in self._cache:
                print(f"[CacheManager] Auto-cleaning up garbage collected item: {key}")
                self._remove_item(key)
    
    def invalidate(self, key: str) -> bool:
        """
        Manually invalidate a cache entry
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if item was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False
    
    def clear(self):
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            self._cache_sizes.clear()
            self._access_times.clear()
            self._weak_refs.clear()
            self._current_memory = 0
            print("[CacheManager] Cache cleared")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics
        
        Returns:
            Dictionary with memory usage info
        """
        with self._lock:
            return {
                "current_memory_mb": self._current_memory / (1024**2),
                "max_memory_mb": self.max_memory / (1024**2),
                "memory_usage_percent": (self._current_memory / self.max_memory) * 100,
                "cached_items": len(self._cache),
                "max_items": self.max_items
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": hit_rate,
            "evictions": self._evictions,
            "total_requests": total_requests
        }
    
    def cleanup_if_needed(self):
        """
        Perform cleanup if memory pressure is detected
        """
        with self._lock:
            # Check system memory pressure
            try:
                free_memory = mm.get_free_memory()
                total_memory = mm.get_total_memory()
                
                if free_memory / total_memory < 0.2:  # Less than 20% free memory
                    print("[CacheManager] Memory pressure detected, performing cleanup")
                    self._evict_oldest_half()
            except Exception as e:
                print(f"[CacheManager] Error checking memory pressure: {e}")
    
    def _evict_oldest_half(self):
        """Evict the oldest half of cached items"""
        items_to_evict = len(self._cache) // 2
        
        if items_to_evict > 0:
            # Sort by access time and evict oldest
            oldest_items = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )[:items_to_evict]
            
            for key, _ in oldest_items:
                if key in self._cache:
                    self._remove_item(key)
                    self._evictions += 1
    
    def get_cached_keys(self) -> list:
        """Get list of currently cached keys"""
        with self._lock:
            return list(self._cache.keys())
    
    def has_key(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            return key in self._cache
    
    def get_item_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a cached item
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary with item info or None if not found
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            return {
                "key": key,
                "size_mb": self._cache_sizes.get(key, 0) / (1024**2),
                "access_time": self._access_times.get(key, 0),
                "age_seconds": time.time() - self._access_times.get(key, 0)
            }
    
    def optimize_cache(self):
        """
        Optimize cache by removing unused items and defragmenting
        """
        with self._lock:
            print("[CacheManager] Optimizing cache...")
            
            # Remove items that have been garbage collected
            dead_refs = []
            for key, weak_ref in self._weak_refs.items():
                if weak_ref() is None:
                    dead_refs.append(key)
            
            for key in dead_refs:
                self._remove_item(key)
            
            # Rebuild ordered dict to defragment
            if self._cache:
                items_by_access = sorted(
                    self._access_times.items(),
                    key=lambda x: x[1],
                    reverse=True  # Most recent first
                )
                
                new_cache = OrderedDict()
                for key, _ in items_by_access:
                    if key in self._cache:
                        new_cache[key] = self._cache[key]
                
                self._cache = new_cache
            
            print(f"[CacheManager] Cache optimized. Items: {len(self._cache)}, "
                  f"Memory: {self._current_memory / (1024**2):.1f}MB")
    
    def set_memory_limit(self, max_memory_gb: float):
        """
        Update memory limit and evict items if necessary
        
        Args:
            max_memory_gb: New memory limit in GB
        """
        with self._lock:
            old_limit = self.max_memory
            self.max_memory = max_memory_gb * (1024**3)
            
            print(f"[CacheManager] Memory limit updated: "
                  f"{old_limit / (1024**3):.1f}GB -> {max_memory_gb:.1f}GB")
            
            # Evict items if we're now over the limit
            if self._current_memory > self.max_memory:
                self._make_space(0)  # This will evict until we're under the limit


class CacheStats:
    """Helper class for cache statistics and monitoring"""
    
    def __init__(self, cache_manager: SmartCacheManager):
        self.cache_manager = cache_manager
    
    def print_summary(self):
        """Print a summary of cache performance"""
        stats = self.cache_manager.get_statistics()
        memory = self.cache_manager.get_memory_usage()
        
        print("\n=== Cache Performance Summary ===")
        print(f"Hit Rate: {stats['hit_rate_percent']:.1f}%")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hits: {stats['hits']}")
        print(f"Cache Misses: {stats['misses']}")
        print(f"Evictions: {stats['evictions']}")
        print(f"Memory Usage: {memory['current_memory_mb']:.1f}MB / {memory['max_memory_mb']:.1f}MB "
              f"({memory['memory_usage_percent']:.1f}%)")
        print(f"Cached Items: {memory['cached_items']} / {memory['max_items']}")
        print("================================\n")
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed cache report"""
        stats = self.cache_manager.get_statistics()
        memory = self.cache_manager.get_memory_usage()
        
        # Get per-item details
        items = []
        for key in self.cache_manager.get_cached_keys():
            item_info = self.cache_manager.get_item_info(key)
            if item_info:
                items.append(item_info)
        
        # Sort by access time (most recent first)
        items.sort(key=lambda x: x['access_time'], reverse=True)
        
        return {
            "performance": stats,
            "memory": memory,
            "items": items,
            "recommendations": self._generate_recommendations(stats, memory)
        }
    
    def _generate_recommendations(self, stats: Dict, memory: Dict) -> list:
        """Generate cache optimization recommendations"""
        recommendations = []
        
        if stats['hit_rate_percent'] < 50:
            recommendations.append("Low hit rate - consider increasing cache size or reviewing usage patterns")
        
        if memory['memory_usage_percent'] > 90:
            recommendations.append("High memory usage - consider reducing cache size or clearing old items")
        
        if stats['evictions'] > stats['hits']:
            recommendations.append("High eviction rate - cache size may be too small for workload")
        
        if memory['cached_items'] < memory['max_items'] // 2:
            recommendations.append("Low cache utilization - consider reducing cache size to free system memory")
        
        return recommendations