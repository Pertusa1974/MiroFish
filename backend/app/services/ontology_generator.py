"""
本体生成服务
接口1：分析文本内容，生成适合社会模拟的实体和关系类型定义
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


ONTOLOGY_SYSTEM_PROMPT = (
    "你是一个专业的知识图谱本体设计专家。你的任务是分析给定的文本内容和模拟需求，"
    "设计适合社交媒体舆论模拟的实体类型和关系类型。\n\n"
    "重要：你必须输出有效的JSON格式数据，不要输出任何其他内容。\n\n"
    "输出JSON格式，包含以下字段：\n"
    "- entity_types: 列表，每项包含 name(PascalCase英文), description(英文,50字符内), "
    "attributes(列表,每项含name/type/description), examples(列表,最多2项)\n"
    "- edge_types: 列表，每项包含 name(UPPER_SNAKE_CASE英文), description(英文,50字符内), "
    "source_targets(列表,每项含source/target), attributes(空列表)\n"
    "- analysis_summary: 字符串，简要分析\n\n"
    "设计规则：\n"
    "1. 必须正好10个实体类型\n"
    "2. 最后2个必须是 Person 和 Organization（兜底类型）\n"
    "3. 前8个是根据文本内容设计的具体类型\n"
    "4. 所有实体必须是现实中可以在社媒上发声的主体\n"
    "5. 属性名不能使用 name, uuid, group_id, created_at, summary\n"
    "6. 每个实体类型只需1个属性\n"
    "7. 关系类型6-8个\n"
    "8. 所有描述保持极简，50字符以内\n"
    "9. examples列表最多2个\n\n"
    "实体类型参考(个人具体类): Student, Professor, Journalist, Celebrity, Executive, Official\n"
    "实体类型参考(组织具体类): University, Company, GovernmentAgency, MediaOutlet, NGO\n"
    "实体类型参考(兜底): Person, Organization\n\n"
    "关系类型参考: WORKS_FOR, AFFILIATED_WITH, REPRESENTS, REGULATES, REPORTS_ON, "
    "COMMENTS_ON, RESPONDS_TO, SUPPORTS, OPPOSES, COLLABORATES_WITH, COMPETES_WITH\n"
)


class OntologyGenerator:

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        lang_instruction = get_language_instruction()
        system_prompt = (
            ONTOLOGY_SYSTEM_PROMPT + "\n\n" + lang_instruction +
            "\nIMPORTANT: Entity type names MUST be in English PascalCase. "
            "Relationship type names MUST be in English UPPER_SNAKE_CASE. "
            "Attribute names MUST be in English snake_case. "
            "Keep ALL descriptions under 50 characters. "
            "Only 1 attribute per entity. Maximum 8 edge types. "
            "Be extremely concise to stay within token limits."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=8192
        )

        result = self._validate_and_process(result)

        return result

    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(原文共{original_length}字，已截取前{self.MAX_TEXT_LENGTH_FOR_LLM}字用于本体分析)..."

        message = f"## 模拟需求\n\n{simulation_requirement}\n\n## 文档内容\n\n{combined_text}\n"

        if additional_context:
            message += f"\n## 额外说明\n\n{additional_context}\n"

        message += (
            "\n请根据以上内容，设计适合社会舆论模拟的实体类型和关系类型。\n\n"
            "CRITICAL RULES:\n"
            "1. Output exactly 10 entity types\n"
            "2. Last 2 must be: Person and Organization\n"
            "3. First 8 are specific types based on document\n"
            "4. All entities must be real-world social media actors\n"
            "5. NO reserved attribute names: name, uuid, group_id, created_at, summary\n"
            "6. Only 1 attribute per entity type\n"
            "7. Maximum 8 edge types\n"
            "8. Keep descriptions under 50 characters\n"
            "9. Maximum 2 examples per entity\n"
            "10. Be extremely concise to stay within token limits\n"
        )

        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        entity_name_map = {}
        for entity in result["entity_types"]:
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        for edge in result["edge_types"]:
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        person_fallback = {
            "name": "Person",
            "description": "Any individual not fitting other person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name"}
            ],
            "examples": ["ordinary citizen"]
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any org not fitting other organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Organization name"}
            ],
            "examples": ["community group"]
        }

        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            if current_count + needed_slots > MAX_ENTITY_TYPES:
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                result["entity_types"] = result["entity_types"][:-to_remove]

            result["entity_types"].extend(fallbacks_to_add)

        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        code_lines = [
            '"""',
            '自定义实体类型定义',
            '由MiroFish自动生成，用于社会舆论模拟',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== 实体类型定义 ==============',
            '',
        ]

        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        code_lines.append('# ============== 关系类型定义 ==============')
        code_lines.append('')

        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        code_lines.append('# ============== 类型配置 ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')

        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')

        return '\n'.join(code_lines)
