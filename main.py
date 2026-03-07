"""
main.py
연설 분석 시스템 CLI 진입점.

실행: python main.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# main.py 위치를 기준으로 절대 경로 설정
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_files"
os.chdir(BASE_DIR)

# .env 파일 로드
load_dotenv(BASE_DIR / ".env")

console = Console()

# ── 메인 메뉴 ──────────────────────────────────────────────────────────────


def show_banner() -> None:
    banner = Text()
    banner.append("연설 분석 시스템", style="bold cyan")
    banner.append("  v1.0", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))


def main_menu() -> None:
    while True:
        console.print()
        table = Table(box=box.ROUNDED, show_header=False, border_style="cyan", padding=(0, 2))
        table.add_column(style="bold yellow", no_wrap=True)
        table.add_column(style="white")
        table.add_row("1", "연설 파일 분석 및 Notion 업로드")
        table.add_row("0", "종료")
        console.print(table)

        choice = Prompt.ask("\n[bold]메뉴 선택[/bold]", choices=["0", "1"], default="0")

        if choice == "0":
            console.print("[dim]종료합니다.[/dim]")
            sys.exit(0)
        elif choice == "1":
            menu_analyze()


# ── 메뉴 1: 연설 분석 & 업로드 ────────────────────────────────────────────


def menu_analyze() -> None:
    from extractor import list_input_files, get_text_from_file, list_jw_outlines
    from analyzer import analyze_speech
    from notion_db import create_talk

    console.print()
    console.rule("[bold cyan]연설 파일 분석 및 Notion 업로드[/bold cyan]")

    # 파일 목록 표시
    files = list_input_files(str(INPUT_DIR))
    if not files:
        console.print("[yellow]input_files 폴더에 파일이 없습니다.[/yellow]")
        console.print(f"  경로: [dim]{INPUT_DIR}[/dim]")
        console.print("  오디오(mp3/m4a/wav) 또는 PDF/TXT 파일을 위 폴더에 넣어주세요.")
        return

    # 연설 파일 선택
    console.print("\n[bold]📂 input_files 파일 목록:[/bold]")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("#", style="cyan", width=4)
    t.add_column("파일명")
    t.add_column("유형", style="dim", width=8)
    for i, f in enumerate(files, 1):
        t.add_row(str(i), f["name"], f["type"])
    console.print(t)

    speech_idx = _pick_index("연설 파일 번호", len(files))
    if speech_idx is None:
        return
    speech_file = files[speech_idx]

    # 골자 파일 선택
    pdf_files = [f for f in files if f["name"].lower().endswith(".pdf")]
    outline_path: str = ""
    outline_page_index: Optional[int] = None
    outline_number: Optional[int] = None

    if pdf_files:
        console.print("\n[bold]📋 골자(Outline) PDF 선택:[/bold]")
        t2 = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        t2.add_column("#", style="cyan", width=4)
        t2.add_column("파일명")
        for i, f in enumerate(pdf_files, 1):
            t2.add_row(str(i), f["name"])
        console.print(t2)
        outline_idx = _pick_index("골자 파일 번호", len(pdf_files))
        if outline_idx is None:
            return
        selected_pdf = pdf_files[outline_idx]
        outline_path = selected_pdf["path"]

        # 파일명에서 골자 번호 자동 추출 (예: S-34_KO_001.pdf → 1)
        m = re.search(r'_(\d{3})\.[^.]+$', selected_pdf["name"])
        if m:
            outline_number = int(m.group(1))
            console.print(f"\n[cyan]✓ 골자 번호 {outline_number:03d} 자동 인식됨[/cyan]")
        else:
            # 자동 추출 실패 → JW 모음집 여부 확인 후 수동 입력
            with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                          console=console, transient=True) as p:
                t = p.add_task("골자 파일 분석 중...", total=None)
                jw_map = list_jw_outlines(outline_path)

            if jw_map:
                min_n, max_n = min(jw_map), max(jw_map)
                console.print(
                    f"\n[cyan]JW 골자 모음집 감지됨[/cyan] — "
                    f"골자 번호 {min_n}~{max_n}번 ({len(jw_map)}개)"
                )
                while True:
                    raw = Prompt.ask(f"[bold]사용할 골자 번호[/bold] ({min_n}~{max_n}, [dim]0=취소[/dim])")
                    if raw == "0":
                        console.print("[dim]취소하고 메뉴로 돌아갑니다.[/dim]")
                        return
                    if raw.isdigit() and int(raw) in jw_map:
                        outline_number = int(raw)
                        outline_page_index = jw_map[outline_number]
                        break
                    console.print(f"[red]{min_n}~{max_n} 범위의 번호를 입력하세요.[/red]")
            else:
                raw = Prompt.ask("[bold]골자 번호 직접 입력[/bold] (없으면 Enter 건너뜀)", default="")
                if raw.strip().isdigit():
                    outline_number = int(raw.strip())
    else:
        console.print("[yellow]PDF 골자 파일이 없습니다. 골자 없이 분석합니다.[/yellow]")

    # 메타데이터 입력
    console.print()
    speaker = Prompt.ask("[bold]연설자 이름[/bold] ([dim]0=취소[/dim])")
    if speaker.strip() == "0":
        console.print("[dim]취소하고 메뉴로 돌아갑니다.[/dim]")
        return
    date_str = Prompt.ask("[bold]날짜[/bold] (YYYY-MM-DD)", default="")
    source_type = "audio" if speech_file["type"] == "오디오" else "pdf"

    # ── 처리 시작 ──
    console.print()

    speech_text: str = ""
    outline_text: str = ""
    analysis: dict = {}
    page_url: str = ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:

        # 1. 연설 텍스트 추출
        task1 = progress.add_task("연설 텍스트 추출 중...", total=None)
        try:
            speech_text = get_text_from_file(speech_file["path"])
        except Exception as e:
            progress.stop()
            console.print(f"[red]오류 - 텍스트 추출 실패:[/red] {e}")
            return
        progress.update(task1, description="[green]✓ 연설 텍스트 추출 완료[/green]")

        # 2. 골자 텍스트 추출 (JW 모음집이면 Vision OCR 사용)
        if outline_path:
            task2_label = (
                "골자 텍스트 추출 중 (Claude Vision OCR)..."
                if outline_page_index is not None
                else "골자 텍스트 추출 중..."
            )
            task2 = progress.add_task(task2_label, total=None)
            try:
                outline_text = get_text_from_file(outline_path, page_index=outline_page_index)
            except Exception as e:
                progress.stop()
                console.print(f"[red]오류 - 골자 추출 실패:[/red] {e}")
                return
            progress.update(task2, description="[green]✓ 골자 텍스트 추출 완료[/green]")

        # 3. Claude 분석
        task3 = progress.add_task("Claude 분석 중...", total=None)
        try:
            analysis = analyze_speech(speech_text, outline_text)
        except Exception as e:
            progress.stop()
            console.print(f"[red]오류 - 분석 실패:[/red] {e}")
            return
        progress.update(task3, description="[green]✓ Claude 분석 완료[/green]")

        # 4. Talks DB 페이지 생성
        task4 = progress.add_task("Notion 연설 페이지 생성 중...", total=None)
        metadata = {
            "speaker": speaker,
            "date": date_str,
            "source": source_type,
            "outline_number": outline_number,
        }
        try:
            page_url = create_talk(metadata, analysis)
        except Exception as e:
            progress.stop()
            console.print(f"[red]오류 - Notion 업로드 실패:[/red] {e}")
            return
        progress.update(task4, description="[green]✓ Notion 업로드 완료[/green]")

    # 결과 출력
    _print_analysis_summary(analysis, page_url)


def _print_analysis_summary(analysis: dict, page_url: str) -> None:
    console.print()
    console.rule("[bold green]분석 결과 요약[/bold green]")

    adherence = analysis.get("outline_adherence", {})
    score = adherence.get("score", "-")
    notes = adherence.get("notes", "")
    search_tags = analysis.get("search_tags", {})
    scriptures = search_tags.get("scriptures", [])
    illustrations = search_tags.get("illustrations", [])
    main_points = analysis.get("structure_and_flow", {}).get("main_points", [])

    summary = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    summary.add_column(style="bold cyan", width=16, no_wrap=True)
    summary.add_column()
    summary.add_row("주제", analysis.get("topic", "-"))
    summary.add_row("골자 일치도", f"{score}점  {notes[:60] + '...' if len(notes) > 60 else notes}")
    summary.add_row("본론 사상 수", f"{len(main_points)}개")
    summary.add_row("주요 성구", f"{len(scriptures)}개")
    summary.add_row("비유 주제", ", ".join(illustrations) if illustrations else "-")
    console.print(summary)

    if page_url:
        console.print()
        console.print(Panel(
            f"[bold green]Notion 업로드 완료![/bold green]\n[link={page_url}]{page_url}[/link]",
            border_style="green",
            padding=(0, 2),
        ))



# ── 공통 유틸 ──────────────────────────────────────────────────────────────


def _pick_index(prompt_label: str, count: int) -> Optional[int]:
    """1-based 번호 입력 → 0-based 인덱스 반환. 0 입력 시 None 반환(취소)."""
    while True:
        raw = Prompt.ask(f"[bold]{prompt_label}[/bold] (1~{count}, [dim]0=취소[/dim])")
        if raw == "0":
            console.print("[dim]취소하고 메뉴로 돌아갑니다.[/dim]")
            return None
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        console.print(f"[red]0~{count} 사이의 번호를 입력하세요.[/red]")


# ── 환경변수 사전 검증 ─────────────────────────────────────────────────────


def _check_env() -> None:
    required = [
        "ANTHROPIC_API_KEY",
        "NOTION_TOKEN",
        "NOTION_TALKS_DB_ID",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        console.print(Panel(
            "[red bold]필수 환경변수가 설정되지 않았습니다.[/red bold]\n\n"
            + "\n".join(f"  • {k}" for k in missing)
            + "\n\n[dim].env 파일을 생성하고 값을 채워주세요. (.env.example 참고)[/dim]",
            border_style="red",
            title="설정 오류",
        ))
        sys.exit(1)


# ── 진입점 ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    show_banner()
    _check_env()
    main_menu()
