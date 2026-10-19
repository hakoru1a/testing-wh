#!/usr/bin/env python3
"""
Quét thư mục input/, transcribe các file MP3 chưa có SRT.
Dùng: python transcribe.py [--model medium] [--lang ja]

Cài: pip install faster-whisper
So sánh với openai-whisper:
  openai-whisper  : whisper.load_model(name)          → model.transcribe(path) → dict
  faster-whisper  : WhisperModel(name, device, compute)→ model.transcribe(path) → (generator, info)
  Ưu điểm faster  : nhanh hơn ~4x, RAM thấp hơn, hỗ trợ int8/float16
"""
import os
import argparse
from faster_whisper import WhisperModel

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

def fmt(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='medium', choices=['tiny','base','small','medium','large','large-v2','large-v3'])
    parser.add_argument('--lang', default='ja')
    # faster-whisper dùng int8 mặc định → nhanh hơn, ít RAM hơn
    parser.add_argument('--compute', default='int8', choices=['int8','float16','float32'])
    args = parser.parse_args()

    audio_exts = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.webm'}
    files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in audio_exts]

    if not files:
        print('❌ Không có file âm thanh nào trong thư mục input/')
        return

    pending = [f for f in files if not os.path.exists(
        os.path.join(INPUT_DIR, os.path.splitext(f)[0] + '.srt'))]

    if not pending:
        print('✅ Tất cả file đã có SRT rồi.')
        return

    print(f'🎙  Load model "{args.model}" ({args.compute})...')
    # device="cpu" tự động, compute_type=int8 nhanh nhất trên CPU
    model = WhisperModel(args.model, device='cpu', compute_type=args.compute)

    for fname in pending:
        fpath = os.path.join(INPUT_DIR, fname)
        base  = os.path.splitext(fname)[0]
        srt_path = os.path.join(INPUT_DIR, base + '.srt')

        print(f'\n⚙️  Transcribing: {fname}')

        # faster-whisper: transcribe() trả về (generator, info) — khác openai-whisper
        segments_gen, info = model.transcribe(fpath, language=args.lang, beam_size=5)

        count = 0
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments_gen, 1):  # seg là object, không phải dict
                f.write(f"{i}\n{fmt(seg.start)} --> {fmt(seg.end)}\n{seg.text.strip()}\n\n")
                count += 1

        print(f'✅ Đã lưu: {base}.srt  ({count} đoạn · lang={info.language})')

if __name__ == '__main__':
    main()
