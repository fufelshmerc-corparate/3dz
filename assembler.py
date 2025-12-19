#!/usr/bin/env python3
"""
Ассемблер для учебной виртуальной машины (УВМ)
"""

import sys
import struct
import argparse
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

class Opcode(Enum):
    """Коды операций УВМ"""
    LOAD = 0  # Чтение значения из памяти
    SHIFT_RIGHT = 2  # Побитовый арифметический сдвиг вправо
    CONST = 3  # Загрузка константы
    STORE = 4  # Запись значения в память

class Instruction:
    """Представление одной инструкции"""
    def __init__(self, opcode: Opcode, operand: int = 0):
        self.opcode = opcode
        self.operand = operand
    
    def to_binary(self) -> bytes:
        """Преобразование инструкции в бинарное представление"""
        if self.opcode == Opcode.CONST:
            operand_masked = self.operand & 0x1FFFFFF 
            value = (operand_masked << 3) | self.opcode.value
            return struct.pack('<I', value)  
            
        elif self.opcode == Opcode.STORE or self.opcode == Opcode.SHIFT_RIGHT:
            operand_masked = self.operand & 0x3FF  
            value = (operand_masked << 3) | self.opcode.value
            return struct.pack('<H', value) 
        elif self.opcode == Opcode.LOAD:
            return struct.pack('B', self.opcode.value)
        
        else:
            raise ValueError(f"Unknown opcode: {self.opcode}")

class Assembler:
    """Ассемблер УВМ"""
    MNEMONICS = {
        'load': Opcode.LOAD,
        'shr': Opcode.SHIFT_RIGHT,
        'const': Opcode.CONST,
        'store': Opcode.STORE,
    }
    def to_binary(self) -> bytes:
        """Преобразование всех инструкций в бинарный формат"""
        binary_data = b''
        for instr in self.instructions:
            binary_data += instr.to_binary()
        return binary_data
    
    def print_internal_representation(self):
        """Вывод внутреннего представления программы в формате полей и значений"""
        print("Internal representation:")
        print("-" * 40)
        
        for i, instr in enumerate(self.instructions):
            print(f"Instruction {i}:")
            if instr.opcode == Opcode.CONST:
                print(f"  A={instr.opcode.value}, B={instr.operand}")
            elif instr.opcode == Opcode.LOAD:
                print(f"  A={instr.opcode.value}")
            elif instr.opcode == Opcode.STORE:
                print(f"  A={instr.opcode.value}, B={instr.operand}")
            elif instr.opcode == Opcode.SHIFT_RIGHT:
                print(f"  A={instr.opcode.value}, B={instr.operand}")
            binary = instr.to_binary()
            hex_bytes = ', '.join([f'0x{byte:02X}' for byte in binary])
            print(f"  Binary: {hex_bytes}")
            print()

    def __init__(self):
        self.instructions: List[Instruction] = []
        self.labels: Dict[str, int] = {}
    
    def parse_line(self, line: str) -> Optional[Tuple[str, List[str]]]:
        """Парсинг строки исходного кода"""
        if ';' in line:
            line = line.split(';')[0]
        
        line = line.strip()
        if not line:
            return None
        if line.endswith(':'):
            label = line[:-1].strip()
            return ('LABEL', [label])
        parts = line.split()
        if not parts:
            return None
        
        mnemonic = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        if args:
            all_args = ' '.join(args).split(',')
            args = [arg.strip() for arg in all_args if arg.strip()]
        
        return (mnemonic, args)
    
    def assemble(self, source_code: str) -> List[Instruction]:
        """Ассемблирование исходного кода"""
        lines = source_code.strip().split('\n')
        self.instructions = []
        self.labels = {}
        address = 0
        for line_num, line in enumerate(lines, 1):
            parsed = self.parse_line(line)
            if not parsed:
                continue
            
            mnemonic, args = parsed
            
            if mnemonic == 'LABEL':
                label = args[0]
                if label in self.labels:
                    raise ValueError(f"Duplicate label '{label}' at line {line_num}")
                self.labels[label] = address
            else:
                if mnemonic not in self.MNEMONICS:
                    raise ValueError(f"Unknown mnemonic '{mnemonic}' at line {line_num}")
                address += self.get_instruction_size(mnemonic)
        address = 0
        for line_num, line in enumerate(lines, 1):
            parsed = self.parse_line(line)
            if not parsed:
                continue
            
            mnemonic, args = parsed
            
            if mnemonic == 'LABEL':
                continue
            
            opcode = self.MNEMONICS[mnemonic]
            operand = 0
            if mnemonic == 'const':
                if len(args) != 1:
                    raise ValueError(f"CONST requires 1 argument at line {line_num}")
                operand = self.parse_operand(args[0], line_num)
            
            elif mnemonic == 'store' or mnemonic == 'shr':
                if len(args) != 1:
                    raise ValueError(f"{mnemonic.upper()} requires 1 argument at line {line_num}")
                operand = self.parse_operand(args[0], line_num)
            
            elif mnemonic == 'load':
                if args:
                    raise ValueError(f"LOAD takes no arguments at line {line_num}")
            
            self.instructions.append(Instruction(opcode, operand))
            address += self.get_instruction_size(mnemonic)
        
        return self.instructions
    
    def parse_operand(self, operand_str: str, line_num: int) -> int:
        """Парсинг операнда (число или метка)"""
        try:
            if operand_str.startswith('0x'):
                return int(operand_str, 16)
            elif operand_str.startswith('0b'):
                return int(operand_str, 2)
            else:
                return int(operand_str)
        except ValueError:
            if operand_str in self.labels:
                return self.labels[operand_str]
            else:
                raise ValueError(f"Unknown label '{operand_str}' at line {line_num}")
    
    def get_instruction_size(self, mnemonic: str) -> int:
        """Получение размера инструкции в байтах"""
        if mnemonic == 'const':
            return 4
        elif mnemonic in ['store', 'shr']:
            return 2
        elif mnemonic == 'load':
            return 1
        else:
            return 0
    
def to_binary(self) -> bytes:
    """Преобразование инструкции в бинарное представление"""
    if self.opcode == Opcode.CONST:
        value = (self.operand << 3) | self.opcode.value
        return struct.pack('<I', value)
    
    elif self.opcode == Opcode.STORE:
        value = (self.operand << 3) | self.opcode.value
        return struct.pack('<H', value)
    
    elif self.opcode == Opcode.SHIFT_RIGHT:
        value = (self.operand << 3) | self.opcode.value
        return struct.pack('<H', value)
    
    elif self.opcode == Opcode.LOAD:
        return struct.pack('B', self.opcode.value)
    
    else:
        raise ValueError(f"Unknown opcode: {self.opcode}")
    
def print_internal_representation(self):
        """Вывод внутреннего представления программы"""
        print("Internal representation:")
        print("-" * 40)
        for i, instr in enumerate(self.instructions):
            print(f"Instruction {i}:")
            print(f"  Opcode: {instr.opcode.name} ({instr.opcode.value})")
            print(f"  Operand: {instr.operand} (0x{instr.operand:X})")
            print(f"  Binary: {instr.to_binary().hex()}")
            print()    

def main():
    parser = argparse.ArgumentParser(description='Assembler for Educational Virtual Machine')
    parser.add_argument('input_file', help='Path to source file')
    parser.add_argument('output_file', help='Path to binary output file')
    parser.add_argument('--test', action='store_true', help='Test mode - show internal representation')
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        assembler = Assembler()
        instructions = assembler.assemble(source_code)
        if args.test:
            assembler.print_internal_representation()
            test_specification_tests()
        

        binary_data = assembler.to_binary()  
        print(f"Successfully assembled {len(instructions)} instructions")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def test_specification_tests():
    """Тестирование примеров из спецификации"""
    print("\nRunning specification tests:")
    
    print("\n1. Testing CONST A=3, B=291 (0x1B, 0x09, 0x00, 0x00):")
    instr = Instruction(Opcode.CONST, 291)
    binary = instr.to_binary()
    expected = bytes([0x1B, 0x09, 0x00, 0x00])
    print(f"   Generated: {binary.hex()}")
    print(f"   Expected:  {expected.hex()}")
    print(f"   Match: {binary == expected}")
    
    print("\n2. Testing LOAD A=0 (0x00):")
    instr = Instruction(Opcode.LOAD)
    binary = instr.to_binary()
    expected = bytes([0x00])
    print(f"   Generated: {binary.hex()}")
    print(f"   Expected:  {expected.hex()}")
    print(f"   Match: {binary == expected}")
    
    print("\n3. Testing STORE A=4, B=163 (0x1C, 0x05):")
    instr = Instruction(Opcode.STORE, 163)
    binary = instr.to_binary()
    expected = bytes([0x1C, 0x05])
    print(f"   Generated: {binary.hex()}")
    print(f"   Expected:  {expected.hex()}")
    print(f"   Match: {binary == expected}")
    
    print("\n4. Testing SHR A=2, B=217 (0xCA, 0x06):")
    instr = Instruction(Opcode.SHIFT_RIGHT, 217)
    binary = instr.to_binary()
    expected = bytes([0xCA, 0x06])
    print(f"   Generated: {binary.hex()}")
    print(f"   Expected:  {expected.hex()}")
    print(f"   Match: {binary == expected}")

if __name__ == '__main__':
    main()