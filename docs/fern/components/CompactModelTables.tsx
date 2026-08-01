/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export interface CompactModelTablesProps {
  children: React.ReactNode;
}

/** Keeps generated model-table metadata columns compact and left-aligned. */
export function CompactModelTables({ children }: CompactModelTablesProps) {
  return (
    <div className="compact-model-tables">
      <style>{`
        .compact-model-tables .fern-table {
          table-layout: auto !important;
        }

        .compact-model-tables .fern-table th,
        .compact-model-tables .fern-table td {
          text-align: left !important;
        }

        .compact-model-tables .fern-table th:nth-child(-n + 2),
        .compact-model-tables .fern-table td:nth-child(-n + 2) {
          width: 1%;
          white-space: nowrap;
        }
      `}</style>
      {children}
    </div>
  );
}
