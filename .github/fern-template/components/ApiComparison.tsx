/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ApiComparisonProps {
  leftTitle: string;
  rightTitle: string;
  leftCode: string;
  rightCode: string;
  language?: string;
}

function CodeCell({ code, language }: { code: string; language?: string }) {
  const codeClassName = language ? `language-${language}` : undefined;

  return (
    <pre
      style={{
        margin: 0,
        overflowX: "auto",
        whiteSpace: "pre",
      }}
    >
      <code className={codeClassName}>{code.trim()}</code>
    </pre>
  );
}

export function ApiComparison({
  leftTitle,
  rightTitle,
  leftCode,
  rightCode,
  language = "python",
}: ApiComparisonProps) {
  return (
    <div style={{ margin: "1rem 0 1.5rem", overflowX: "auto" }}>
      <table style={{ tableLayout: "fixed", width: "100%" }}>
        <thead>
          <tr>
            <th style={{ width: "50%" }}>{leftTitle}</th>
            <th style={{ width: "50%" }}>{rightTitle}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ verticalAlign: "top" }}>
              <CodeCell code={leftCode} language={language} />
            </td>
            <td style={{ verticalAlign: "top" }}>
              <CodeCell code={rightCode} language={language} />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
