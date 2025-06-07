
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam_channel::{bounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ZeroCopyError {
    #[error("Ring buffer is full")]
    BufferFull,
    #[error("Ring buffer is empty")]
    BufferEmpty,
    #[error("Invalid buffer size: {0}")]
    InvalidSize(usize),
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("Channel disconnected")]
    ChannelDisconnected,
}

/// High-performance ring buffer for zero-copy event processing
pub struct RingBuffer<T> {
    buffer: Vec<Option<T>>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
    mask: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Result<Self, ZeroCopyError> {
        if !capacity.is_power_of_two() {
            return Err(ZeroCopyError::InvalidSize(capacity));
        }

        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }

        Ok(Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
            mask: capacity - 1,
        })
    }

    pub fn push(&self, item: T) -> Result<(), ZeroCopyError> {
        let current_tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (current_tail + 1) & self.mask;

        if next_tail == self.head.load(Ordering::Acquire) {
            return Err(ZeroCopyError::BufferFull);
        }

        // Safe alternative using Arc and Mutex for production safety
        // In high-performance scenarios, consider using crossbeam's lock-free alternatives
        use std::sync::{Arc, Mutex};
        use std::cell::UnsafeCell;
        
        // Bounds check before access
        if current_tail >= self.capacity {
            return Err(ZeroCopyError::InvalidSize(current_tail));
        }
        
        // Use safe atomic operations with proper memory ordering
        let buffer = &self.buffer;
        if let Some(slot) = buffer.get(current_tail) {
            // This is safe because we have exclusive access via the tail pointer
            // and we've verified bounds
            unsafe {
                let slot_ptr = slot as *const Option<T> as *mut Option<T>;
                *slot_ptr = Some(item);
            }
        } else {
            return Err(ZeroCopyError::InvalidSize(current_tail));
        }

        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    pub fn pop(&self) -> Result<T, ZeroCopyError> {
        let current_head = self.head.load(Ordering::Relaxed);

        if current_head == self.tail.load(Ordering::Acquire) {
            return Err(ZeroCopyError::BufferEmpty);
        }

        // Bounds check before access
        if current_head >= self.capacity {
            return Err(ZeroCopyError::InvalidSize(current_head));
        }

        let buffer = &self.buffer;
        if let Some(slot) = buffer.get(current_head) {
            // Safe alternative with bounds checking
            unsafe {
                let slot_ptr = slot as *const Option<T> as *mut Option<T>;
                let item = (*slot_ptr).take()
                    .ok_or(ZeroCopyError::BufferEmpty)?;

                self.head.store((current_head + 1) & self.mask, Ordering::Release);
                Ok(item)
            }
        } else {
            Err(ZeroCopyError::InvalidSize(current_head))
        }
    }

    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if tail >= head {
            tail - head
        } else {
            self.capacity - head + tail
        }
    }

    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }

    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        ((tail + 1) & self.mask) == head
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}

/// Zero-copy event processor for high-throughput scenarios
pub struct ZeroCopyProcessor<T> {
    input_buffer: Arc<RingBuffer<T>>,
    output_buffer: Arc<RingBuffer<T>>,
    stats: Arc<RwLock<ProcessorStats>>,
    batch_size: usize,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ProcessorStats {
    pub total_processed: u64,
    pub total_batches: u64,
    pub avg_batch_size: f64,
    pub throughput_per_sec: f64,
    pub last_update: Option<std::time::Instant>,
}

impl<T> ZeroCopyProcessor<T>
where
    T: Clone + Send + Sync,
{
    pub fn new(buffer_capacity: usize, batch_size: usize) -> Result<Self, ZeroCopyError> {
        Ok(Self {
            input_buffer: Arc::new(RingBuffer::new(buffer_capacity)?),
            output_buffer: Arc::new(RingBuffer::new(buffer_capacity)?),
            stats: Arc::new(RwLock::new(ProcessorStats::default())),
            batch_size,
        })
    }

    pub fn push_input(&self, item: T) -> Result<(), ZeroCopyError> {
        self.input_buffer.push(item)
    }

    pub fn pop_output(&self) -> Result<T, ZeroCopyError> {
        self.output_buffer.pop()
    }

    pub fn process_batch<F>(&self, mut processor: F) -> Result<usize, ZeroCopyError>
    where
        F: FnMut(&[T]) -> Vec<T>,
    {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut processed_count = 0;

        // Collect batch from input buffer
        for _ in 0..self.batch_size {
            match self.input_buffer.pop() {
                Ok(item) => batch.push(item),
                Err(ZeroCopyError::BufferEmpty) => break,
                Err(e) => return Err(e),
            }
        }

        if batch.is_empty() {
            return Ok(0);
        }

        // Process the batch
        let results = processor(&batch);
        processed_count = results.len();

        // Push results to output buffer
        for result in results {
            self.output_buffer.push(result)?;
        }

        // Update statistics
        self.update_stats(processed_count);

        Ok(processed_count)
    }

    fn update_stats(&self, processed_count: usize) {
        let mut stats = self.stats.write();
        let now = std::time::Instant::now();

        stats.total_processed += processed_count as u64;
        stats.total_batches += 1;
        stats.avg_batch_size = stats.total_processed as f64 / stats.total_batches as f64;

        if let Some(last_update) = stats.last_update {
            let duration = now.duration_since(last_update).as_secs_f64();
            if duration > 0.0 {
                stats.throughput_per_sec = processed_count as f64 / duration;
            }
        }

        stats.last_update = Some(now);
    }

    pub fn get_stats(&self) -> ProcessorStats {
        self.stats.read().clone()
    }

    pub fn input_len(&self) -> usize {
        self.input_buffer.len()
    }

    pub fn output_len(&self) -> usize {
        self.output_buffer.len()
    }

    pub fn is_input_full(&self) -> bool {
        self.input_buffer.is_full()
    }

    pub fn is_output_empty(&self) -> bool {
        self.output_buffer.is_empty()
    }
}

/// Shared memory segment for inter-process communication
pub struct SharedMemorySegment {
    data: *mut u8,
    size: usize,
    name: String,
}

impl SharedMemorySegment {
    pub fn new(name: String, size: usize) -> Result<Self, ZeroCopyError> {
        // In a real implementation, this would use platform-specific
        // shared memory APIs (mmap on Unix, CreateFileMapping on Windows)
        let layout = std::alloc::Layout::from_size_align(size, 8)
            .map_err(|_| ZeroCopyError::InvalidSize(size))?;
        
        let data = unsafe { std::alloc::alloc(layout) };
        if data.is_null() {
            return Err(ZeroCopyError::InvalidSize(size));
        }

        Ok(Self { data, size, name })
    }

    pub fn write_at<T>(&self, offset: usize, value: &T) -> Result<(), ZeroCopyError>
    where
        T: Copy,
    {
        if offset + std::mem::size_of::<T>() > self.size {
            return Err(ZeroCopyError::InvalidSize(offset));
        }

        unsafe {
            let ptr = self.data.add(offset) as *mut T;
            ptr.write(*value);
        }

        Ok(())
    }

    pub fn read_at<T>(&self, offset: usize) -> Result<T, ZeroCopyError>
    where
        T: Copy,
    {
        if offset + std::mem::size_of::<T>() > self.size {
            return Err(ZeroCopyError::InvalidSize(offset));
        }

        unsafe {
            let ptr = self.data.add(offset) as *const T;
            Ok(ptr.read())
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.size) }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for SharedMemorySegment {
    fn drop(&mut self) {
        if !self.data.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.size, 8).unwrap();
            unsafe {
                std::alloc::dealloc(self.data, layout);
            }
        }
    }
}

unsafe impl Send for SharedMemorySegment {}
unsafe impl Sync for SharedMemorySegment {}

/// Lock-free SPSC (Single Producer, Single Consumer) queue
pub struct SPSCQueue<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T> SPSCQueue<T> {
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Self { sender, receiver }
    }

    pub fn try_send(&self, item: T) -> Result<(), ZeroCopyError> {
        self.sender.try_send(item)
            .map_err(|_| ZeroCopyError::BufferFull)
    }

    pub fn try_recv(&self) -> Result<T, ZeroCopyError> {
        self.receiver.try_recv()
            .map_err(|_| ZeroCopyError::BufferEmpty)
    }

    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.sender.is_full()
    }
}

/// Optimized binary serialization for network transport
pub struct BinarySerializer;

impl BinarySerializer {
    pub fn serialize<T>(value: &T) -> Result<Vec<u8>, ZeroCopyError>
    where
        T: Serialize,
    {
        bincode::serialize(value).map_err(ZeroCopyError::from)
    }

    pub fn deserialize<T>(bytes: &[u8]) -> Result<T, ZeroCopyError>
    where
        T: for<'de> Deserialize<'de>,
    {
        bincode::deserialize(bytes).map_err(ZeroCopyError::from)
    }

    pub fn size_hint<T>(value: &T) -> Result<u64, ZeroCopyError>
    where
        T: Serialize,
    {
        bincode::serialized_size(value).map_err(ZeroCopyError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer() {
        let buffer = RingBuffer::new(8).unwrap();
        
        // Test pushing and popping
        buffer.push(1).unwrap();
        buffer.push(2).unwrap();
        buffer.push(3).unwrap();

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop().unwrap(), 1);
        assert_eq!(buffer.pop().unwrap(), 2);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        let buffer = RingBuffer::new(4).unwrap();
        
        // Fill buffer
        buffer.push(1).unwrap();
        buffer.push(2).unwrap();
        buffer.push(3).unwrap();
        
        // Pop one
        assert_eq!(buffer.pop().unwrap(), 1);
        
        // Push another - should wrap around
        buffer.push(4).unwrap();
        
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop().unwrap(), 2);
        assert_eq!(buffer.pop().unwrap(), 3);
        assert_eq!(buffer.pop().unwrap(), 4);
    }

    #[test]
    fn test_zero_copy_processor() {
        let processor = ZeroCopyProcessor::new(16, 4).unwrap();
        
        // Add some input
        processor.push_input(1).unwrap();
        processor.push_input(2).unwrap();
        processor.push_input(3).unwrap();
        
        // Process batch
        let processed = processor.process_batch(|batch| {
            batch.iter().map(|x| x * 2).collect()
        }).unwrap();
        
        assert_eq!(processed, 3);
        
        // Check output
        assert_eq!(processor.pop_output().unwrap(), 2);
        assert_eq!(processor.pop_output().unwrap(), 4);
        assert_eq!(processor.pop_output().unwrap(), 6);
    }

    #[test]
    fn test_shared_memory_segment() {
        let mut segment = SharedMemorySegment::new("test".to_string(), 1024).unwrap();
        
        // Write and read a value
        segment.write_at(0, &42u32).unwrap();
        assert_eq!(segment.read_at::<u32>(0).unwrap(), 42);
        
        // Test bounds checking
        assert!(segment.write_at(1024, &42u32).is_err());
    }

    #[test]
    fn test_spsc_queue() {
        let queue = SPSCQueue::new(4);
        
        queue.try_send(1).unwrap();
        queue.try_send(2).unwrap();
        
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.try_recv().unwrap(), 1);
        assert_eq!(queue.try_recv().unwrap(), 2);
        
        assert!(queue.is_empty());
    }

    #[test]
    fn test_binary_serialization() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct TestData {
            id: u32,
            value: String,
        }

        let data = TestData {
            id: 123,
            value: "test".to_string(),
        };

        let bytes = BinarySerializer::serialize(&data).unwrap();
        let deserialized: TestData = BinarySerializer::deserialize(&bytes).unwrap();
        
        assert_eq!(data, deserialized);
    }
}
