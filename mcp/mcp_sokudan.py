#!/usr/bin/env python3
"""
Sokudan MCP Server for Codex CLI
Codex CLI から sokudan（日本語 System One 意思決定モデル）を呼び出せる MCP サーバー
"""

import json
from mcp.server.fastmcp import FastMCP

# sokudan は遅延ロード（サーバー起動を速するため）
agent = None

def get_agent():
    global agent
    if agent is None:
        import sokudan
        agent = sokudan.load("GeneLab/sokudan-ja-310m")
    return agent

mcp = FastMCP("sokudan")


@mcp.tool()
async def analyze_japanese_text(
    text: str,
    question_type: str = "choice",
    instructions: str = "",
    criteria: str = "",
) -> str:
    """
    日本語テキストを分析し、指定した型付き質問に対する回答と確率を返す。

    Args:
        text: 分析対象の日本語テキスト
        question_type: 質問タイプ（"choice", "score", "bool" のいずれか）
        instructions: 質問の指示文（例: "この問い合わせはどの部署が担当すべきか"）
        criteria: 選択肢定義（JSON文字列。choiceの場合は {"ラベル": "説明", ...}、
                  scoreの場合は ["低", "中", "高"] のような配列、bool不要）

    Returns:
        JSON文字列で回答結果を返す
    """
    state = {"body": text}

    try:
        parsed_criteria = json.loads(criteria) if criteria else {}
    except json.JSONDecodeError:
        return json.dumps(
            {"error": "criteria は有効な JSON 文字列である必要があります"},
            ensure_ascii=False,
        )

    if question_type not in ("choice", "score", "bool", "noul"):
        return json.dumps(
            {"error": f"不明な question_type: {question_type}"},
            ensure_ascii=False,
        )

    if question_type == "noul":
        question_type = "bool"

    question_def = {
        "type": question_type,
        "instructions": instructions,
    }
    if criteria:
        question_def["criteria"] = parsed_criteria

    questions = {"analysis": question_def}

    try:
        agent = get_agent()
        result = agent.predict(state, questions)
        answer = result["answers"]["analysis"]

        output = {}
        if question_type == "choice":
            output["choice"] = answer.get("choice")
            output["confidence"] = answer.get("confidence")
            output["probabilities"] = answer.get("probs")
        elif question_type == "score":
            output["score"] = answer.get("score")
            output["confidence"] = answer.get("confidence")
            output["distribution"] = answer.get("dist")
        elif question_type in ("bool", "noul"):
            output["probability_true"] = answer.get("noul")
            output["confidence"] = answer.get("confidence")

        return json.dumps(output, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def route_japanese_email(
    body: str,
    departments: str = '{"billing": "支払い・返金", "technical": "不具合・障害", "sales": "料金・新規契約", "other": "上記以外"}',
) -> str:
    """
    日本語のメール本文から最適な担当部署を判定する。

    Args:
        body: メール本文
        departments: 部署定義（JSON文字列 {"ラベル": "説明", ...}）

    Returns:
        担当部署と確率を JSON で返す
    """
    try:
        deps = json.loads(departments)
    except json.JSONDecodeError:
        return json.dumps(
            {"error": "departments は有効な JSON 文字列である必要があります"},
            ensure_ascii=False,
        )

    state = {"body": body}
    questions = {
        "department": {
            "type": "choice",
            "instructions": "この問い合わせはどの部署が担当すべきか",
            "criteria": deps,
        }
    }

    try:
        agent = get_agent()
        result = agent.predict(state, questions)
        answer = result["answers"]["department"]
        return json.dumps({
            "department": answer.get("choice"),
            "confidence": answer.get("confidence"),
            "probabilities": answer.get("probs"),
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def score_urgency(
    text: str,
    levels: str = '["急がない", "早めに", "業務が止まっている"]',
) -> str:
    """
    日本語テキストの緊急度を順序尺度でスコアリングする。

    Args:
        text: 分析対象のテキスト
        levels: 緊急度の段階定義（JSON列文字列）

    Returns:
        スコアと確率分布を JSON で返す
    """
    try:
        parsed_levels = json.loads(levels)
    except json.JSONDecodeError:
        return json.dumps(
            {"error": "levels は有効な JSON 配列文字列である必要があります"},
            ensure_ascii=False,
        )

    state = {"body": text}
    questions = {
        "urgency": {
            "type": "score",
            "instructions": "この依頼の緊急度は",
            "criteria": parsed_levels,
        }
    }

    try:
        agent = get_agent()
        result = agent.predict(state, questions)
        answer = result["answers"]["urgency"]
        return json.dumps({
            "score": answer.get("score"),
            "max_score": len(parsed_levels) - 1,
            "confidence": answer.get("confidence"),
            "distribution": answer.get("dist"),
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def check_bool(text: str, question: str) -> str:
    """
    日本語テキストに対して YES/NO 質問を投げ、確率を返す。

    Args:
        text: 分析対象のテキスト
        question: YES/NO で答える質問文（例: "解約を示唆しているか"）

    Returns:
        P(True) と確信度を JSON で返す
    """
    state = {"body": text}
    questions = {
        "check": {
            "type": "bool",
            "instructions": question,
        }
    }

    try:
        agent = get_agent()
        result = agent.predict(state, questions)
        answer = result["answers"]["check"]
        return json.dumps({
            "question": question,
            "probability_true": answer.get("noul"),
            "confidence": answer.get("confidence"),
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
