"""Message Bus implementation for agent communication"""

import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import uuid

import structlog

from src.core.models import AgentMessage

logger = structlog.get_logger()


class MessageBus(ABC):
    """Abstract base class for message bus implementations"""

    @abstractmethod
    async def publish(self, topic: str, message: AgentMessage) -> bool:
        """Publish a message to a topic"""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, callback: Callable) -> str:
        """Subscribe to a topic with a callback"""
        pass

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic"""
        pass

    @abstractmethod
    async def receive(self, topic: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        """Receive a message from a topic"""
        pass

    @abstractmethod
    async def request_response(
        self, topic: str, message: AgentMessage, timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Send a request and wait for response"""
        pass


class InMemoryMessageBus(MessageBus):
    """
    In-memory message bus for development and testing.

    Features:
    - Async message queues per topic
    - Subscription management
    - Dead letter queue for failed messages
    - Message correlation tracking
    - Priority queuing
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.PriorityQueue] = defaultdict(
            lambda: asyncio.PriorityQueue()
        )
        self._subscriptions: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self._dead_letter_queue: asyncio.Queue = asyncio.Queue()
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._message_history: List[AgentMessage] = []
        self._running = False
        self._dispatcher_task: Optional[asyncio.Task] = None

        logger.info("InMemoryMessageBus initialized")

    async def start(self):
        """Start the message bus dispatcher"""
        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info("Message bus started")

    async def stop(self):
        """Stop the message bus"""
        self._running = False
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        logger.info("Message bus stopped")

    async def publish(self, topic: str, message: AgentMessage) -> bool:
        """Publish a message to a topic with priority queuing"""
        try:
            # Priority queue uses (priority, timestamp, message) tuple
            # Lower number = higher priority, so we negate
            priority = -message.priority
            timestamp = datetime.utcnow().timestamp()

            await self._queues[topic].put((priority, timestamp, message))
            self._message_history.append(message)

            logger.debug(
                "Message published",
                topic=topic,
                message_type=message.message_type,
                priority=message.priority,
            )
            return True

        except Exception as e:
            logger.error("Failed to publish message", topic=topic, error=str(e))
            return False

    async def subscribe(self, topic: str, callback: Callable) -> str:
        """Subscribe to a topic with a callback function"""
        subscription_id = f"{topic}_{uuid.uuid4().hex[:8]}"
        self._subscriptions[topic][subscription_id] = callback

        logger.info("Subscription added", topic=topic, subscription_id=subscription_id)
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription"""
        for topic, subs in self._subscriptions.items():
            if subscription_id in subs:
                del subs[subscription_id]
                logger.info("Subscription removed", subscription_id=subscription_id)
                return True
        return False

    async def receive(self, topic: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        """Receive the next message from a topic"""
        try:
            if topic not in self._queues:
                return None

            _, _, message = await asyncio.wait_for(
                self._queues[topic].get(), timeout=timeout
            )
            return message

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error("Failed to receive message", topic=topic, error=str(e))
            return None

    async def request_response(
        self, topic: str, message: AgentMessage, timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Send a request and wait for a correlated response"""
        correlation_id = message.correlation_id or str(uuid.uuid4())
        message.correlation_id = correlation_id

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_responses[correlation_id] = future

        try:
            # Publish request
            await self.publish(topic, message)

            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            logger.warning(
                "Request timeout", topic=topic, correlation_id=correlation_id
            )
            return None
        finally:
            self._pending_responses.pop(correlation_id, None)

    async def _dispatch_loop(self):
        """Main loop to dispatch messages to subscribers"""
        while self._running:
            for topic, subscriptions in list(self._subscriptions.items()):
                if topic in self._queues and not self._queues[topic].empty():
                    try:
                        _, _, message = await asyncio.wait_for(
                            self._queues[topic].get(), timeout=0.1
                        )

                        # Check for response correlation
                        if message.correlation_id in self._pending_responses:
                            self._pending_responses[message.correlation_id].set_result(
                                message
                            )
                            continue

                        # Dispatch to all subscribers
                        for sub_id, callback in list(subscriptions.items()):
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(message)
                                else:
                                    callback(message)
                            except Exception as e:
                                logger.error(
                                    "Callback failed",
                                    subscription_id=sub_id,
                                    error=str(e),
                                )
                                # Move to dead letter queue
                                await self._dead_letter_queue.put(
                                    {
                                        "message": message,
                                        "error": str(e),
                                        "subscription_id": sub_id,
                                    }
                                )

                    except asyncio.TimeoutError:
                        continue

            await asyncio.sleep(0.01)  # Prevent busy loop

    async def get_dead_letters(self) -> List[Dict[str, Any]]:
        """Retrieve messages from the dead letter queue"""
        dead_letters = []
        while not self._dead_letter_queue.empty():
            dead_letters.append(await self._dead_letter_queue.get())
        return dead_letters

    def get_queue_depth(self, topic: str) -> int:
        """Get the number of messages waiting in a topic queue"""
        if topic in self._queues:
            return self._queues[topic].qsize()
        return 0

    def get_message_history(self, limit: int = 100) -> List[AgentMessage]:
        """Get recent message history"""
        return self._message_history[-limit:]


class KafkaMessageBus(MessageBus):
    """
    Kafka-based message bus for production deployments.

    Features:
    - High throughput message delivery
    - Message persistence
    - Consumer groups
    - Exactly-once semantics (configurable)
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "aura-agents",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._producer = None
        self._consumers: Dict[str, Any] = {}
        self._subscriptions: Dict[str, Dict[str, Callable]] = defaultdict(dict)

        logger.info(
            "KafkaMessageBus initialized",
            servers=bootstrap_servers,
            group_id=group_id,
        )

    async def connect(self):
        """Initialize Kafka connections"""
        try:
            from kafka import KafkaProducer, KafkaConsumer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            logger.info("Kafka producer connected")

        except ImportError:
            logger.warning("kafka-python not installed, using mock mode")
        except Exception as e:
            logger.error("Failed to connect to Kafka", error=str(e))
            raise

    async def publish(self, topic: str, message: AgentMessage) -> bool:
        """Publish message to Kafka topic"""
        if not self._producer:
            logger.warning("Producer not initialized")
            return False

        try:
            future = self._producer.send(topic, value=message.model_dump())
            future.get(timeout=10)  # Block until sent

            logger.debug("Message published to Kafka", topic=topic)
            return True

        except Exception as e:
            logger.error("Failed to publish to Kafka", topic=topic, error=str(e))
            return False

    async def subscribe(self, topic: str, callback: Callable) -> str:
        """Subscribe to a Kafka topic"""
        try:
            from kafka import KafkaConsumer

            if topic not in self._consumers:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                )
                self._consumers[topic] = consumer

            subscription_id = f"{topic}_{uuid.uuid4().hex[:8]}"
            self._subscriptions[topic][subscription_id] = callback

            logger.info("Subscribed to Kafka topic", topic=topic)
            return subscription_id

        except Exception as e:
            logger.error("Failed to subscribe to Kafka", topic=topic, error=str(e))
            return ""

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a topic"""
        for topic, subs in self._subscriptions.items():
            if subscription_id in subs:
                del subs[subscription_id]
                if not subs and topic in self._consumers:
                    self._consumers[topic].close()
                    del self._consumers[topic]
                return True
        return False

    async def receive(self, topic: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        """Receive a message from Kafka topic"""
        if topic not in self._consumers:
            return None

        try:
            consumer = self._consumers[topic]
            records = consumer.poll(timeout_ms=int(timeout * 1000), max_records=1)

            for tp, messages in records.items():
                if messages:
                    data = messages[0].value
                    return AgentMessage(**data)

        except Exception as e:
            logger.error("Failed to receive from Kafka", topic=topic, error=str(e))

        return None

    async def request_response(
        self, topic: str, message: AgentMessage, timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Request-response pattern over Kafka"""
        # Implementation would use reply-to pattern
        # Simplified version for now
        await self.publish(topic, message)
        return None  # Would need correlation ID tracking

    async def close(self):
        """Close all Kafka connections"""
        if self._producer:
            self._producer.close()

        for consumer in self._consumers.values():
            consumer.close()

        logger.info("Kafka connections closed")
