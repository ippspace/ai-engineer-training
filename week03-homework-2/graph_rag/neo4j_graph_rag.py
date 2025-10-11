import json
from dataclasses import dataclass
from typing import LiteralString, cast

from neo4j import Driver
from neo4j_graphrag.llm import OpenAILLM

from .llama_index_rag import LlamaIndexRAG


@dataclass
class Entity:
    name: str
    type: str


@dataclass
class Relationship:
    source: str
    target: str
    type: str


class Neo4jGraphRAG:
    def __init__(
        self, driver: Driver, llm: OpenAILLM, index_rag: LlamaIndexRAG, data_path: str
    ):
        self.driver: Driver = driver
        self.llm: OpenAILLM = llm
        self.index_rag: LlamaIndexRAG = index_rag

        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
            self.build_graph(text)

    def __del__(self):
        """析构函数，确保正确关闭Neo4j驱动"""
        if hasattr(self, "driver") and self.driver:
            try:
                self.driver.close()
            except Exception:
                # 忽略关闭时的异常，避免干扰程序退出
                pass

    def close(self):
        """显式关闭Neo4j驱动"""
        self.__del__()

    def query(self, question: str, company: str = "A公司", top_k: int = 3) -> str:
        """查询"""
        retrievals = self.index_rag.retrieve(question, top_k=top_k)
        index_context = ""
        if retrievals:
            index_context = "\n".join(
                [
                    f"{i + 1}(score={r.score:.4f}). {r.text}"
                    for i, r in enumerate(retrievals)
                ]
            )
        graph_context = ""
        relationships = self.query_shareholder_relationship(company)
        if relationships:
            graph_context = "\n".join(
                [f"{r.source} -> {r.type} -> {r.target}" for r in relationships]
            )

        # 生成答案
        prompt = f"""
        基于`索引上下文`和`公司股权关系图谱`回答问题：

        索引上下文:
        {index_context}

        公司股权关系图谱:
        {graph_context}
        
        请简洁问题：{question}
        """

        response = self.llm.invoke(prompt)
        result = f"""
问题：{question}
回答：{response.content.strip()}

{"=" * 30} 文档检索 {"=" * 30}
{index_context}

{"=" * 30} 公司股权关系图谱 {"=" * 30}
{graph_context}
"""
        return result

    def extract_entities(self, text: str) -> list[Entity]:
        """提取公司股东实体"""
        prompt = f"""
        从文本中提取公司股东实体，包括个人股东和机构股东，及员工持股平台。
        
        文本：{text}
        
        直接返回有效的JSON格式字符串（不包含任何额外文本，比如 markdown 格式）：
        {{
            "entities": [
                {{"name": "股东名称", "type": "股东类型"}}
            ]
        }}
        """

        response = self.llm.invoke(prompt)
        result = json.loads(response.content)

        entities = [Entity(e["name"], e["type"]) for e in result.get("entities", [])]
        print(f"提取到 {len(entities)} 个公司股东")
        for e in entities:
            print(f"    • {e.name} ({e.type})")
        print()
        return entities

    def extract_relationships(
        self,
        text: str,
        entities: list[Entity],
    ) -> list[Relationship]:
        """提取公司股东持股比例关系"""
        entity_names = [e.name for e in entities]

        prompt = f"""
        从文本中提取公司股东的持股比例：
        
        文本：{text}
        公司股东：{entity_names}
        
        将提取到的股东持股比例，填入JSON格式的"type"字段中，格式为： "持股x%"。
        直接返回有效的JSON格式字符串（不包含任何额外文本，比如 markdown 格式）：
        {{
            "relationships": [
                {{"source": "股东名称", "target": "公司名称", "type": "持股x%"}}
            ]   
        }}
        """

        response = self.llm.invoke(prompt)
        result = json.loads(response.content)

        relationships: list[Relationship] = []
        for r in result.get("relationships", []):
            relationships.append(Relationship(r["source"], r["target"], r["type"]))

        print(f"提取到 {len(relationships)} 个公司股东持股比例关系")
        for r in relationships:
            print(f"    • {r.source} -> {r.type} -> {r.target}")
        print()
        return relationships

    def build_graph(self, text: str):
        entities: list[Entity] = self.extract_entities(text)
        relationships: list[Relationship] = self.extract_relationships(text, entities)

        """构建公司股东图谱"""
        with self.driver.session() as session:
            # 清空现有数据
            _ = session.run("MATCH (n) DETACH DELETE n")

            # 写入股东实体
            for entity in entities:
                # 使用参数化查询，避免直接在字符串中拼接标签名
                query = f"MERGE (n:{entity.type} {{name: $name}})"
                _ = session.run(cast(LiteralString, query), name=entity.name)

            # 确保目标公司节点存在（从relationships中提取所有唯一的target）
            companies = set(rel.target for rel in relationships)
            for company in companies:
                # 添加公司节点，使用Company类型
                query = "MERGE (n:Company {name: $name})"
                _ = session.run(cast(LiteralString, query), name=company)

            # 写入股东关系
            for rel in relationships:
                # 对于包含特殊字符的关系类型，使用反引号转义
                query = f"""
                MATCH (a {{name: $source}})
                MATCH (b {{name: $target}})
                MERGE (a)-[:`{rel.type}`]->(b)
                """
                _ = session.run(
                    cast(LiteralString, query),
                    source=rel.source,
                    target=rel.target,
                )

        print("公司股东关系图谱构建完成")

    def query_shareholder_relationship(
        self, company_name: str = "A公司", shareholder_name: str | None = None
    ) -> list[Relationship]:
        """查询指定股东与公司的持股关系

        Args:
            company_name: 公司名称，默认为"A公司"
            shareholder_name: 股东名称，空字符串或None时查询所有股东

        Returns:
            包含持股关系信息的列表，每个元素为一个Relationship对象
        """
        with self.driver.session() as session:
            # 根据shareholder_name参数决定查询逻辑
            if not shareholder_name:
                # 查询所有与该公司相关的持股关系
                query = cast(
                    LiteralString,
                    """
                    MATCH (shareholder)-[r]->(company {name: $company_name})
                    RETURN shareholder.name AS source, 
                           company.name AS target, 
                           type(r) AS relationship_type
                    """,
                )
                result = session.run(query, company_name=company_name)
            else:
                # 查询特定股东与公司之间的直接持股关系
                query = cast(
                    LiteralString,
                    """
                    MATCH (shareholder {name: $shareholder_name})-[r]->(company {name: $company_name})
                    RETURN shareholder.name AS source, 
                           company.name AS target, 
                           type(r) AS relationship_type
                    """,
                )
                result = session.run(
                    query,
                    shareholder_name=shareholder_name,
                    company_name=company_name,
                )

            records = list(result)

            if not records:
                return []

            # 提取关系信息
            return [
                Relationship(
                    record["source"],
                    record["target"],
                    record["relationship_type"],
                )
                for record in records
            ]
