"""Learning Agent - Continuous improvement and adaptation"""

import json
from datetime import datetime
from typing import Any, Dict, List

import structlog
from langchain_core.tools import Tool

from src.core.base_agent import BaseAgent
from src.core.models import (
    AgentConfig,
    KnowledgeItem,
    KnowledgeType,
    TaskRequest,
    TaskResponse,
)

from src.db import SessionLocal
from src.models.orm import ORMLearning, ORMStrategy

logger = structlog.get_logger()


class LearningAgent(BaseAgent):
    """
    Agent responsible for continuous learning and system improvement.

    Capabilities:
    - Reinforcement learning from audit outcomes
    - Policy effectiveness tracking
    - Test quality improvement
    - Model behavior learning
    - Knowledge base maintenance
    """
    print('Learning Agent')
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="learning",
                llm_model="gpt-4",
                llm_provider="openai",
            )
        super().__init__(config)

        self._learning_history: List[Dict[str, Any]] = []
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._performance_metrics: Dict[str, List[float]] = {}
        
        # Load from DB
        self._load_from_db()

        logger.info("LearningAgent initialized with Database backend", agent_id=self.id)

    def _load_from_db(self):
        """Load learning history and strategies from database"""
        db = SessionLocal()
        try:
            # Load learning history (last 100)
            history = db.query(ORMLearning).order_by(ORMLearning.timestamp.desc()).limit(100).all()
            for item in history:
                self._learning_history.append({
                    "audit_id": item.audit_id,
                    "insights": item.insights,
                    "timestamp": item.timestamp.isoformat()
                })
            # Reverse to restore chronological order if needed, assuming append adds to end
            self._learning_history.reverse()

            # Load strategies
            strategies = db.query(ORMStrategy).all()
            for strategy in strategies:
                self._strategies[strategy.type] = {
                    "updated_at": strategy.updated_at.isoformat(),
                    "optimization": strategy.optimization
                }

        except Exception as e:
            logger.error("Failed to load learning data", error=str(e))
        finally:
            db.close()

    def _init_tools(self) -> List[Any]:
        """Initialize learning-specific tools"""
        return [
            Tool(
                name="optimize_strategy",
                func=self._optimize_strategy_sync,
                description="Optimize a strategy using reinforcement learning",
            ),
            Tool(
                name="extract_knowledge",
                func=self._extract_knowledge_sync,
                description="Extract knowledge from experiences",
            ),
        ]

    async def process_task(self, task: TaskRequest) -> TaskResponse:
        """Process learning-related tasks"""
        start_time = datetime.utcnow()

        try:
            if task.task_type == "learn_from_audit":
                result = await self._learn_from_audit(task.parameters)
            elif task.task_type == "optimize_strategies":
                result = await self._optimize_strategies(task.parameters)
            elif task.task_type == "update_knowledge":
                result = await self._update_knowledge_base(task.parameters)
            elif task.task_type == "evaluate_effectiveness":
                result = await self._evaluate_effectiveness(task.parameters)
            elif task.task_type == "transfer_learning":
                result = await self._transfer_learning(task.parameters)
            else:
                result = await self._handle_generic_task(task)

            self.metrics.tasks_completed += 1

            return TaskResponse(
                task_id=task.task_id,
                status="success",
                result=result,
                agent_id=self.id,
                execution_time_ms=int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
            )

        except Exception as e:
            self.metrics.tasks_failed += 1
            logger.error("Task failed", task_id=task.task_id, error=str(e))

            return TaskResponse(
                task_id=task.task_id,
                status="failure",
                error=str(e),
                agent_id=self.id,
                execution_time_ms=int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                ),
            )

    async def _learn_from_audit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from audit results"""
        audit_id = params.get("audit_id")
        results = params.get("results", [])
        policies = params.get("policies", [])

        prompt = f"""Analyze this audit to extract learning insights:

Audit ID: {audit_id}
Results: {len(results)} test results
Sample: {json.dumps(results[:10])}
Policies: {json.dumps(policies)}

Extract:
1. Successful patterns to reinforce
2. Failure patterns to avoid
3. Policy improvements needed
4. Test effectiveness insights
5. Model behavior patterns

For each learning:
- Type (reinforcement/correction/optimization)
- Domain
- Confidence score
- Actionable recommendation

Return as JSON."""

        response = await self.invoke_llm(prompt)

        # Store learning
        learning_entry = {
            "audit_id": audit_id,
            "timestamp": datetime.utcnow().isoformat(),
            "insights": response,
        }
        self._learning_history.append(learning_entry)

        # Persist to DB
        db = SessionLocal()
        try:
            orm_learning = ORMLearning(
                audit_id=audit_id,
                insights=response,
                timestamp=datetime.utcnow()
            )
            db.add(orm_learning)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("Failed to persist learning", error=str(e))
        finally:
            db.close()

        # Share with other agents
        knowledge = KnowledgeItem(
            id=f"learning_{datetime.utcnow().timestamp()}",
            knowledge_type=KnowledgeType.EXPERIENCE,
            domain="audit_learning",
            content=learning_entry,
            source_agent=self.name,
            confidence=0.8,
        )
        await self.share_knowledge(knowledge, ["policy", "testing", "audit"])

        return {
            "audit_id": audit_id,
            "learnings_extracted": response,
            "results_analyzed": len(results),
        }

    async def _optimize_strategies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize agent strategies using RL principles"""
        strategy_type = params.get("strategy_type", "testing")
        current_performance = params.get("performance_metrics", {})
        constraints = params.get("constraints", {})

        prompt = f"""Optimize {strategy_type} strategy using reinforcement learning principles:

Current Performance: {json.dumps(current_performance)}
Constraints: {json.dumps(constraints)}
Historical Performance: {len(self._performance_metrics.get(strategy_type, []))} data points

Apply optimization:
1. Q-learning for action selection
2. Policy gradient for continuous improvement
3. Multi-armed bandit for exploration/exploitation

Suggest:
- Updated strategy parameters
- Expected improvement
- Confidence level
- A/B test recommendations
- Rollback criteria"""

        response = await self.invoke_llm(prompt)

        # Update strategy
        self._strategies[strategy_type] = {
        }

        # Persist strategy
        db = SessionLocal()
        try:
            # Merge/Update
            strategy = db.query(ORMStrategy).filter(ORMStrategy.type == strategy_type).first()
            if strategy:
                strategy.optimization = response
                strategy.updated_at = datetime.utcnow()
            else:
                strategy = ORMStrategy(
                    type=strategy_type,
                    optimization=response,
                    updated_at=datetime.utcnow()
                )
                db.add(strategy)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("Failed to persist strategy", error=str(e))
        finally:
            db.close()

        return {
            "strategy_type": strategy_type,
            "optimization": response,
        }

    async def _update_knowledge_base(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update the knowledge base with new learnings"""
        new_knowledge = params.get("knowledge", [])
        prune_old = params.get("prune_old", True)

        stored_count = 0

        for item in new_knowledge:
            if self.knowledge_base:
                knowledge_item = KnowledgeItem(
                    id=f"kb_{datetime.utcnow().timestamp()}_{stored_count}",
                    knowledge_type=KnowledgeType(item.get("type", "experience")),
                    domain=item.get("domain", "general"),
                    content=item.get("content", {}),
                    source_agent=self.name,
                    confidence=item.get("confidence", 0.8),
                    tags=item.get("tags", []),
                )
                await self.knowledge_base.store(knowledge_item)
                stored_count += 1

        # Prune outdated knowledge
        pruned_count = 0
        if prune_old and self.knowledge_base:
            # Would implement decay mechanism here
            pass

        return {
            "items_stored": stored_count,
            "items_pruned": pruned_count,
        }

    async def _evaluate_effectiveness(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate effectiveness of strategies and learnings"""
        evaluation_type = params.get("type", "overall")
        time_period = params.get("time_period_days", 30)

        prompt = f"""Evaluate effectiveness of {evaluation_type}:

Learning History: {len(self._learning_history)} entries
Strategies: {list(self._strategies.keys())}
Time Period: {time_period} days

Evaluate:
1. Strategy performance trends
2. Learning quality metrics
3. Knowledge utilization rate
4. Improvement velocity
5. Areas needing attention

Provide:
- Effectiveness scores by category
- Trend analysis
- Recommendations for improvement
- ROI of learning investments"""

        response = await self.invoke_llm(prompt)

        return {
            "evaluation_type": evaluation_type,
            "effectiveness_report": response,
        }

    async def _transfer_learning(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transfer learning across domains"""
        source_domain = params.get("source_domain")
        target_domain = params.get("target_domain")

        prompt = f"""Apply transfer learning from {source_domain} to {target_domain}:

Analyze what can be transferred:
1. Successful patterns
2. Test strategies
3. Policy optimizations
4. Risk indicators

Adapt for target domain:
- Necessary modifications
- Potential risks
- Expected effectiveness
- Validation approach"""

        response = await self.invoke_llm(prompt)

        return {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "transfer_recommendations": response,
        }

    async def _handle_generic_task(self, task: TaskRequest) -> Dict[str, Any]:
        """Handle generic learning tasks"""
        prompt = f"""Process this learning task:

Task Type: {task.task_type}
Description: {task.description}
Parameters: {task.parameters}"""

        response = await self.invoke_llm(prompt)
        return {"response": response}

    async def _update_strategies(self, experience: Dict[str, Any]):
        """Self-improvement based on own experiences"""
        pass

    def _optimize_strategy_sync(self, strategy: str) -> str:
        return "Strategy optimized"

    def _extract_knowledge_sync(self, data: str) -> str:
        return "Knowledge extracted"

    def get_learning_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get learning history"""
        return self._learning_history[-limit:]

    def get_strategy(self, strategy_type: str) -> Dict[str, Any]:
        """Get a specific strategy"""
        return self._strategies.get(strategy_type, {})
