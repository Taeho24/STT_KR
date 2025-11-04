#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트 모델 정리 스크립트
- test_models/ 폴더 삭제
- HuggingFace 캐시 정리 (선택적)
"""

import argparse
import shutil
from pathlib import Path

def cleanup_test_models(dry_run=False):
    """test_models 폴더 삭제"""
    test_dir = Path("test_models")
    
    if not test_dir.exists():
        print("✅ test_models 폴더가 존재하지 않습니다.")
        return
    
    # 크기 계산
    total_size = sum(f.stat().st_size for f in test_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print(f"\n📁 test_models 폴더")
    print(f"   크기: {size_mb:.1f} MB")
    print(f"   파일 수: {len(list(test_dir.rglob('*')))}")
    
    if dry_run:
        print("\n⚠️  DRY RUN 모드: 실제로 삭제하지 않습니다.")
        print("   삭제할 파일:")
        for item in sorted(test_dir.rglob('*'))[:20]:  # 처음 20개만
            print(f"     - {item.relative_to(test_dir)}")
        if len(list(test_dir.rglob('*'))) > 20:
            print(f"     ... 외 {len(list(test_dir.rglob('*'))) - 20}개")
    else:
        confirm = input(f"\n❓ {size_mb:.1f} MB를 삭제하시겠습니까? (y/N): ")
        if confirm.lower() == 'y':
            shutil.rmtree(test_dir)
            print("✅ test_models 폴더가 삭제되었습니다.")
        else:
            print("❌ 취소되었습니다.")

def cleanup_huggingface_cache(models_to_keep=None, dry_run=False):
    """HuggingFace 캐시 정리 (선택한 모델 제외)"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    if not cache_dir.exists():
        print("\n✅ HuggingFace 캐시가 존재하지 않습니다.")
        return
    
    # 캐시 크기 계산
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    size_gb = total_size / (1024 * 1024 * 1024)
    
    print(f"\n📦 HuggingFace 캐시")
    print(f"   위치: {cache_dir}")
    print(f"   크기: {size_gb:.2f} GB")
    
    if models_to_keep:
        print(f"\n⚠️  다음 모델은 유지됩니다:")
        for model in models_to_keep:
            print(f"     - {model}")
        
        # 실제 구현은 복잡하므로 경고만 표시
        print("\n💡 HuggingFace 캐시는 수동으로 정리하세요:")
        print(f"   rm -rf {cache_dir}/hub")
        print("   (최종 모델 선택 후 권장)")
    else:
        if dry_run:
            print("\n⚠️  DRY RUN 모드")
        else:
            confirm = input(f"\n❓ 전체 캐시 {size_gb:.2f} GB를 삭제하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                shutil.rmtree(cache_dir)
                print("✅ HuggingFace 캐시가 삭제되었습니다.")
            else:
                print("❌ 취소되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="테스트 모델 정리")
    parser.add_argument("--dry-run", action="store_true", help="실제 삭제하지 않고 미리보기")
    parser.add_argument("--all", action="store_true", help="HuggingFace 캐시도 포함")
    parser.add_argument("--keep-models", nargs="+", help="유지할 모델 목록 (HuggingFace)")
    
    args = parser.parse_args()
    
    print("🧹 테스트 모델 정리 스크립트")
    print("=" * 80)
    
    # test_models 폴더 정리
    cleanup_test_models(dry_run=args.dry_run)
    
    # HuggingFace 캐시 정리 (선택적)
    if args.all:
        cleanup_huggingface_cache(models_to_keep=args.keep_models, dry_run=args.dry_run)
    
    print("\n" + "=" * 80)
    print("✅ 정리 완료!")
    
    if not args.dry_run:
        print("\n💡 최종 모델 선택 후 HuggingFace 캐시도 정리하세요:")
        print("   python tools/cleanup.py --all --keep-models <최종_모델_이름>")

if __name__ == "__main__":
    main()
