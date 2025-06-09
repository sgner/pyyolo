#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os

def fix_duplicates():
    try:
        print("开始修复重复方法定义...")
        
        if not os.path.exists('pyqt_ver_app.py'):
            print("错误：找不到pyqt_ver_app.py文件")
            return
        
        with open('pyqt_ver_app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"原文件行数: {len(content.split('\n'))}")
        
        lines = content.split('\n')
        fixed_lines = []
        
        # 记录要跳过的行号范围
        skip_ranges = [
            (1252, 1257),  # 第一个show_overlay和hide_overlay
            (1259, 1280),  # 简单的dragEnterEvent、dropEvent等
        ]
        
        skipped_count = 0
        for i, line in enumerate(lines):
            should_skip = False
            for start, end in skip_ranges:
                if start <= i + 1 <= end:  # +1因为行号从1开始
                    should_skip = True
                    skipped_count += 1
                    break
            
            if not should_skip:
                # 修复overlay引用
                if 'self.overlay' in line and not line.strip().startswith('#'):
                    line = '        # ' + line.strip() + '  # overlay已废弃'
                fixed_lines.append(line)
        
        print(f"跳过了 {skipped_count} 行")
        print(f"修复后行数: {len(fixed_lines)}")
        
        # 写入修复后的文件
        with open('pyqt_ver_app_fixed.py', 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print("修复完成：已删除重复的方法定义并注释掉overlay引用")
        print("生成文件：pyqt_ver_app_fixed.py")
        
    except Exception as e:
        print(f"修复过程中出现错误: {e}")

if __name__ == "__main__":
    fix_duplicates() 